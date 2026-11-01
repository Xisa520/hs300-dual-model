import os,json,random,argparse
import numpy as np,pandas as pd,torch,joblib
from sklearn.metrics import accuracy_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix
from core import indicators,make_images,FEATURES,CNNStock

def metrics(y,p):
    y=np.array(y,dtype=int);pred=(np.asarray(p)>=.5).astype(int)
    return {'n':int(len(y)),'accuracy':round(float(accuracy_score(y,pred)),4),'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),4),'mcc':round(float(matthews_corrcoef(y,pred)),4),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run(csv,epochs=12):
    torch.set_num_threads(2);torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
    d=indicators(pd.read_csv(csv));valid=d[FEATURES].notna().all(axis=1)&d.y_next.notna()
    ids=np.where(valid)[0];n=len(ids);a=ids[int(n*.65)];b=ids[int(n*.8)]
    # 次日标签只需保证训练样本的下一日不落入验证/测试段
    train=(valid & (d.index<a-1));val=(valid & (d.index>=a)&(d.index<b-1));test=(valid & (d.index>=b)&(d.index<len(d)-1))
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=2000,random_state=2026))
    model.fit(d.loc[train,FEATURES],d.loc[train,'y_next'].astype(int))
    out={'data_start':str(d.date.min().date()),'data_end':str(d.date.max().date()),'split_train_end':str(d.loc[train,'date'].max().date()),'split_val_start':str(d.loc[val,'date'].min().date()),'split_test_start':str(d.loc[test,'date'].min().date())}
    out['dma']={'validation':metrics(d.loc[val,'y_next'],model.predict_proba(d.loc[val,FEATURES])[:,1]),'test':metrics(d.loc[test,'y_next'],model.predict_proba(d.loc[test,FEATURES])[:,1])}
    os.makedirs('models',exist_ok=True);joblib.dump(model,'models/dma_classifier.joblib')
    X,idx=make_images(d)
    tr=np.flatnonzero(np.isin(idx,np.where(train)[0]));va=np.flatnonzero(np.isin(idx,np.where(val)[0]));te=np.flatnonzero(np.isin(idx,np.where(test)[0]))
    labels=d.y_next.fillna(0).to_numpy(dtype=np.float32)
    cnn=CNNStock();opt=torch.optim.Adam(cnn.parameters(),lr=.001,weight_decay=.001);loss=torch.nn.BCEWithLogitsLoss()
    best=float('inf');best_state=None;patience=0
    for epoch in range(epochs):
        cnn.train();order=np.random.permutation(tr)
        for st in range(0,len(order),64):
            rows=order[st:st+64];xb=torch.from_numpy(X[rows]);yb=torch.from_numpy(labels[idx[rows]])
            opt.zero_grad();l=loss(cnn(xb),yb);l.backward();opt.step()
        cnn.eval()
        with torch.no_grad():
            v=float(loss(cnn(torch.from_numpy(X[va])),torch.from_numpy(labels[idx[va]])).item())
        if v<best-1e-4:
            best=v;best_state={k:v.detach().clone() for k,v in cnn.state_dict().items()};patience=0
        else:
            patience+=1
            if patience>=4:break
    cnn.load_state_dict(best_state);cnn.eval();torch.save(cnn.state_dict(),'models/cnn_stock.pt')
    with torch.no_grad():
        for key,ix in [('validation',va),('test',te)]:
            p=torch.sigmoid(cnn(torch.from_numpy(X[ix]))).numpy()
            out.setdefault('cnn',{})[key]=metrics(labels[idx[ix]],p)
    out['cnn']['trained_epochs']=epoch+1;out['notes']='复现结构的重新训练实现，不是上一组训练时保存的模型权重；旧报告的2013-2020结果不可与本样本直接比较。'
    with open('models/metrics.json','w',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/hs300_2021_2026.csv');p.add_argument('--epochs',type=int,default=12);args=p.parse_args();run(args.csv,args.epochs)
