docs = [
    "World leaders at the UN Climate Summit finalized a landmark pact today, committing to reduce carbon emissions by 45% before 2030. The agreement includes $100 billion in annual funding for developing nations transitioning to renewable energy. UN Secretary-General António Guterres stated: 'This is a turning point for planetary survival.' Critics, however, warn that enforcement mechanisms remain weak.",
    "Apple's iPhone 17 sold 2 million units within 24 hours of its release, surpassing all previous records. The new model features holographic display technology and extended battery life. According to CEO Tim Cook, 'Demand reflects our commitment to revolutionary innovation.' Analysts project sales may boost Q3 revenue by 20%.",
    "A delegation of 50 Chinese students arrived at Imperial College London for a two-week cultural exchange. Activities included joint robotics workshops and debates on sustainable development. Dr. Emily Roberts, program director, emphasized: 'Such initiatives bridge educational gaps globally.' Plans to double participation in 2026 were announced.",
    "Rescue teams have evacuated 10,000 residents after a 7.1-magnitude quake struck Hokkaido yesterday. The government allocated ¥5 billion ($35 million) for emergency aid, while volunteers distributed food and medical supplies. 'We prioritize restoring infrastructure within 72 hours,' said Prime Minister Fumio Kishida. International support is en route.",
    "NASA's Perseverance rover confirmed traces of liquid water on Phobos, Mars' largest moon. Data suggests subsurface ice deposits could support future manned missions. Chief scientist Dr. Lisa Yang declared: 'This reshapes our understanding of extraterrestrial resources.' Further analysis will determine potential for life-sustaining habitats.",
    "A nationwide digital health service launched today, enabling remote consultations via AI-driven platforms. The system aims to reduce hospital wait times by 30%. Health Secretary Sajid Javid noted: 'Technology democratizes medical access.' Critics raised data privacy concerns, urging stricter safeguards.",
    "India's Ranthambore National Park reported a 25% increase in Bengal tiger populations this year, attributed to anti-poaching drones and community patrols. 'We've relocated 15 cubs to safe zones,' said conservationist Priya Sharma. The success model will be replicated across Southeast Asia.",
    "Iceland now generates 80% of its electricity from geothermal and wind sources, exceeding its 2025 green energy target. Government spokesperson Eva Magnúsdóttir credited 'strategic investments in volcanic heat capture.' The nation aims for full carbon neutrality by 2030.",
    "MIT researchers unveiled an AI system that identifies early-stage lung cancer with 95% accuracy—20% higher than traditional methods. Dr. Alan Turing, project lead, explained: 'Machine learning analyzes scans in seconds.' Hospitals in 10 countries will pilot the technology next month.",
    "Post-Olympic facilities in Paris will become public sports academies for underprivileged youth. Mayor Anne Hidalgo confirmed: 'Legacy matters more than gold medals.' The project includes free training programs and mentorship by retired athletes, benefiting 5,000 teenagers annually."
]

import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig

class FeatureFusionModel(nn.Module):
    def __init__(self, fusion_strategy='early', local_bert_path='./bert_local/'):
        super(FeatureFusionModel, self).__init__()
        self.fusion_strategy = fusion_strategy
        
        # 加载本地BERT模型
        self.config = BertConfig.from_pretrained(local_bert_path)
        self.bert = BertModel.from_pretrained(local_bert_path, config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(local_bert_path)
        
        bert_dim = self.config.hidden_size  # 使用配置中的隐藏层维度
        
        # 主题特征和情感特征的维度（根据实际情况调整）
        topic_dim = 100
        sentiment_dim = 50
        
        # 根据融合策略构建网络
        if fusion_strategy == 'early':
            # 早期融合
            self.fc = nn.Sequential(
                nn.Linear(bert_dim + topic_dim + sentiment_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        elif fusion_strategy == 'attention':
            # 注意力融合
            self.topic_attention = CrossModalAttention(bert_dim, topic_dim)
            self.sentiment_attention = CrossModalAttention(bert_dim, sentiment_dim)
            self.fc = nn.Sequential(
                nn.Linear(bert_dim, 256),  # 注意这里只使用文本特征维度
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # 分类层
        self.classifier = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, topic_features, sentiment_features):
        # BERT文本特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]  # 取[CLS]标记
        
        if self.fusion_strategy == 'early':
            # 直接拼接所有特征
            combined = torch.cat([text_features, topic_features, sentiment_features], dim=1)
            fused = self.fc(combined)
        elif self.fusion_strategy == 'attention':
            # 注意力对齐特征
            topic_aligned = self.topic_attention(text_features, topic_features)
            sentiment_aligned = self.sentiment_attention(text_features, sentiment_features)
            # 特征融合
            fused = self.fc(text_features + topic_aligned + sentiment_aligned)
        
        # 分类输出
        logits = self.classifier(fused)
        return self.softmax(logits)


class CrossModalAttention(nn.Module):
    """跨模态注意力模块（文本特征对齐其他模态特征）"""
    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        
        # 线性变换层
        self.query = nn.Linear(text_dim, text_dim)
        self.key = nn.Linear(feature_dim, text_dim)
        self.value = nn.Linear(feature_dim, text_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.tensor(text_dim, dtype=torch.float32))

    def forward(self, text, feature):
        # 确保特征维度正确
        if feature.size(1) != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {feature.size(1)}")
        
        # 生成Q, K, V
        Q = self.query(text)  # [batch, text_dim]
        K = self.key(feature)  # [batch, text_dim]
        V = self.value(feature)  # [batch, text_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(-1)) / self.scale
        attention_weights = torch.softmax(attention_scores.squeeze(-1), dim=1)
        
        # 加权融合 (batch_size, 1, text_dim) -> (batch_size, text_dim)
        aligned_feature = torch.matmul(attention_weights.unsqueeze(1), V).squeeze(1)
        return aligned_feature


# 本地模型加载函数（确保模型文件存在）
def load_local_bert(local_path):
    """
    验证并加载本地BERT模型
    返回配置、模型和分词器
    """
    # 检查必要文件
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    missing = [f for f in required_files if not os.path.exists(os.path.join(local_path, f))]
    
    if missing:
        raise FileNotFoundError(f"Missing BERT files: {', '.join(missing)}")
    
    config = BertConfig.from_pretrained(local_path)
    model = BertModel.from_pretrained(local_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(local_path)
    
    return config, model, tokenizer


# 示例用法 -------------------------------------------------
if __name__ == "__main__":
    # 设置本地BERT模型路径（替换为实际路径）
    LOCAL_BERT_PATH = "./bert_local/"
    
    # 初始化模型
    model = FeatureFusionModel(
        fusion_strategy='attention',  # 可选 'early' 或 'attention'
        local_bert_path=LOCAL_BERT_PATH
    )
    
    # 模拟输入数据
    input_text = ["This product works well but is expensive."]
    topic_features = torch.randn(1, 100)  # 主题特征
    sentiment_features = torch.randn(1, 50)  # 情感特征
    
    # 使用本地tokenizer分词
    inputs = model.tokenizer(
        input_text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length=128  # 设置最大长度
    )
    
    print("模型输入 IDs:", inputs["input_ids"].shape)
    print("注意力掩码:", inputs["attention_mask"].shape)
    
    # 前向传播
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        topic_features=topic_features,
        sentiment_features=sentiment_features
    )
    
    print("\n预测概率分布:", outputs)

    # 测试预测结果
    predicted_class = torch.argmax(outputs, dim=1)
    print("预测类别:", "正向" if predicted_class.item() == 1 else "负向")