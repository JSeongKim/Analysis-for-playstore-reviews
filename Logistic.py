# -*- coding: utf-8 -*
import os
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split

IP_topics = ['재화 부족 현상', '서버 접속 및 사전 예약 보상', '결제 문제', 
            '콘텐츠', '방탄소년단', '원작 경험', '운영', '재화 밸런스', 
            '뽑기 확률', '과금 문제','원작 캐릭터', '계정 연동', 
            '자동 사냥 및 버그', '업데이트', '잦은 팅김', '과금 유도', 
            '원작 반영', '네트워크 오류']
NoneIP_topics = ['서버 접속 문제', '사전 예약 보상', '캐릭터', '접속 오류', 
                '어려운 난이도', '신규 유저', '와이파이 & 데이터 문제', '고객 센터 불만',
                '결제 & 환불', '레드 와이파이', '밸런스', '과도한 과금 유도', 
                '그래픽 & 이펙트 & 연출', '뽑기확률', '과장/ 허위 광고',
                '매칭 시스템', '브롤스타즈(특정 게임)', '게임 칭찬']

def searchFiles(path):
    filelist = []
    filenames = os.listdir(path)
    for filename in filenames:
        file_path = os.path.join(path, filename)
        filelist.append(file_path)
    return filelist

def main():
    #리뷰 읽기
    reviews = []
    for filePath in searchFiles('./Reviews/IP/'):
        review = pd.read_csv(filePath, encoding = 'utf-8')
        reviews.append(review)
    docs = pd.concat(reviews, ignore_index=True)

    #'PN'필드: 리뷰 평점이 4 이상이면 1(긍정) 3 이하면 0(부정)
    docs['PN'] = docs['평점'].apply(lambda x: 1 if x > 3 else 0) 

    tdm = joblib.load('Data/LDA_IP.pkl')#학습된 LDA 모델 불러옴
    df = pd.DataFrame(tdm, columns=IP_topics)
    
    max_dict = dict()
    for idx, vec in enumerate(tdm):
        t = -1 
        for i, r in enumerate(vec):
            if(r > 0.7):
                t = i
                break
        if(t == -1 or len(docs['내용'][idx])<100 ):
            continue
        if(t not in max_dict):
            max_dict[t] = [idx]
        else:
            max_dict[t].append(idx)
            max_dict[t] = sorted(max_dict[t], key = lambda x: x, reverse=True)

    sorted_review = sorted(max_dict.items(), key = lambda x: x[0], reverse=False)
    
    for key, value in sorted_review:
        print('[주제 {}의 대표 리뷰 ]'.format(IP_topics[key]))
        for v in value[:5]:
            print(docs['내용'][v]+'\n')

    return None
    df = df.drop(columns=['서버 접속 문제', '브롤스타즈(특정 게임)'])#p 값이 큰 변수 제외

    #학습, 검증 셋 분리
    x_train, x_test, y_train, y_test = train_test_split(df, docs['PN'], train_size = 0.7, shuffle = True)

    #로지스틱 회귀
    log = LogisticRegression()
    log.fit(x_train, y_train)

    y_pred = log.predict(x_test)
    print(classification_report(y_test, y_pred))

    logit = sm.Logit(docs['PN'], df)
    result = logit.fit()
    print(result.summary())
    print(np.exp(result.params))#오즈비

    #RUC 곡선 그리기
    fpr, tpr, threshold = roc_curve(y_train, log.decision_function(x_train))

    # plt.plot(fpr, tpr, 'o-', ms=2, label="Logistic Regression")
    # plt.legend()
    # plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    # plt.xlabel('특이도(Specificity')
    # plt.ylabel('재현률(Recall)')
    # plt.title('ROC 커브')
    # plt.show()
    print('AUC :{}'.format(auc(fpr, tpr)))#AUC 값

    return None

if __name__=='__main__':
    main()