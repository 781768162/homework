from LLM_API import extract_columns
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from gensim import corpora
from gensim.models import LdaModel
from openai import OpenAI
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

#nltk.download('stopwords', quiet=True)
#nltk.download('wordnet', quiet=True)
#nltk.download('averaged_perceptron_tagger', quiet=True)
#nltk.download('punkt', quiet=True)
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

def preprocess_text(text):
    # 去除非字母字符并转为小写
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # 分词
    tokens = word_tokenize(cleaned_text)
    
    # 加载停用词集
    stop_words = set(stopwords.words('english'))
    # 移除停用词和短词
    filtered_tokens = [word for word in tokens 
                      if word not in stop_words and len(word) > 2]
    
    # 词性标注辅助获取词形还原的POS标签
    def get_wordnet_pos(treebank_tag):
        tag = treebank_tag[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = pos_tag(filtered_tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag))
        for token, tag in tagged_tokens
    ]
    
    return lemmatized_tokens

def preprocess_documents(documents):
    return [preprocess_text(doc) for doc in documents]

def generate_wordclouds(lda_model, num_topics, num_words=20):
    plt.figure(figsize=(15, 10))
    for i in range(num_topics):
        # 获取主题的关键词及其权重
        topic_words = dict(lda_model.show_topic(i, topn=num_words))
        
        # 生成词云
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            max_words=num_words
        ).generate_from_frequencies(topic_words)
        
        # 绘制词云
        plt.subplot((num_topics+1)//2, 2, i+1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('topic_wordclouds.png')
    plt.show()

def plot_heatmap(lda_model, corpus, document_names=None):
    doc_topics = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    topic_matrix = np.array([[prob for _, prob in topic_dist] for topic_dist in doc_topics])
    
    # 创建DataFrame
    if document_names is None:
        document_names = [f'Document {i+1}' for i in range(len(corpus))]
    
    topic_names = [f'Topic {i+1}' for i in range(topic_matrix.shape[1])]
    df = pd.DataFrame(topic_matrix, index=document_names, columns=topic_names)
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Topic Probability'})
    plt.title('Document-Topic Distribution')
    plt.tight_layout()
    plt.savefig('topic_distribution_heatmap.png')
    plt.show()
    
    return df

def analyze_topics_with_llm(lda_model, num_topics, num_words=10):
    topic_descriptions = []
    for i in range(num_topics):
        topic_words = [word for word, _ in lda_model.show_topic(i, topn=num_words)]
        topic_str = ", ".join(topic_words)
        
        prompt = f"""
        你是一个专业的主题分析助手。请根据以下关键词描述这个主题的内容和含义：
        关键词: {topic_str}
        请用1-2句话概括这个主题的核心内容，并说明它可能涉及哪些领域或应用场景。
        """
        client = OpenAI(
        base_url = "http://localhost:11434/v1",
        api_key = "ollama"
        )
        # 调用OpenAI API
        try:
            response = client.chat.completions.create(
                model="qwen3:4b",
                messages=[
                    {"role": "system", "content": "你是一个专业的主题分析助手。/no_think"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            topic_descriptions.append(analysis)
            print(f"\n主题 {i+1} 分析:")
            print(analysis)
        except Exception as e:
            print(f"主题 {i+1} 分析失败: {str(e)}")
            topic_descriptions.append("分析失败")
    
    return topic_descriptions

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

def main():
    #documents = extract_columns()
    #documents = documents[:10]
    #预处理
    processed_docs = preprocess_documents(docs)
    
    #构建词典
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    print(f"词典大小: {len(dictionary)}")
    
    #生成语料库
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    #训练LDA模型
    num_topics = 4
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15,
        alpha='auto'
    )
    
    #pyLDAvis交互图
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    
    #词云图
    generate_wordclouds(lda_model, num_topics)
    
    #热力图
    topic_dist_df = plot_heatmap(lda_model, corpus, None)
    
    #使用大模型分析主题
    topic_analyses = analyze_topics_with_llm(lda_model, num_topics)
    
    # 保存分析结果
    with open('topic_analysis.txt', 'w') as f:
        for i, analysis in enumerate(topic_analyses):
            f.write(f"主题 {i+1} 分析:\n")
            f.write(analysis + "\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()
