"""Trains PriceLens quantile models on the listing datasets in ../data and writes ../model/model.json.
Run: python3 ml/train.py   (needs pandas numpy scikit-learn lightgbm)"""
import json, re, sys, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
D=os.path.join(os.path.dirname(__file__),'..','data'); OUT=os.path.join(os.path.dirname(__file__),'..','model'); os.makedirs(OUT,exist_ok=True)
REF_YEAR=2020  # newest listing year present in both datasets = data vintage
OWN={'First Owner':1,'Second Owner':2,'Third Owner':3,'Fourth & Above Owner':4,'Test Drive Car':0,'1st owner':1,'2nd owner':2,'3rd owner':3,'4th owner':4}
def num(s):
    m=re.search(r'[\d.]+',str(s)); return float(m.group()) if m else np.nan
def load_cars():
    d=pd.read_csv(f'{D}/cars_v3.csv').drop_duplicates()
    PRE=['Maruti Swift Dzire','Maruti Wagon R','Maruti Alto K10','Maruti Alto 800','Maruti S Cross','Maruti Grand Vitara','Hyundai Grand i10','Hyundai Santro Xing','Hyundai i20','Hyundai Eon','Toyota Land Cruiser','Toyota Etios Liva','Toyota Innova Crysta','Mahindra XUV500','Mahindra TUV 300','Mahindra Xylo','Mahindra E Verito','Tata Indigo eCS','Tata New Safari','Honda City','Honda Brio','Ford Figo','Ford EcoSport','Land Rover Range','Nissan Micra','Renault Kwid','Volkswagen Polo','Volkswagen Vento','Chevrolet Beat','Chevrolet Spark','Chevrolet Sail','Mercedes-Benz E-Class','Mercedes-Benz C-Class','Mercedes-Benz GLA','BMW 3','BMW 5','Audi A4','Audi A6','Skoda Rapid','Skoda Octavia','Skoda Superb','Skoda Fabia']
    def fam(n):
        n=re.sub(r'\s+',' ',n.strip())
        for p in sorted(PRE,key=len,reverse=True):
            if n.startswith(p+' ') or n==p: return p
        t=n.split(); return ' '.join(t[:2])
    def mk(n):
        t=n.split(); return ' '.join(t[:2]) if t[0] in ('Land','Force','Ashok') else t[0]
    d['make']=d.name.map(mk); d['model']=d.name.map(fam)
    d['variant']=d.name
    d['engine_cc']=d.engine.map(num); d['power']=d.max_power.map(num); d['seats']=d.seats.astype(float); d['mileage_k']=d.mileage.map(num)
    d['owner_n']=d.owner.map(OWN); d=d[d.owner_n>0]
    d['price']=d.selling_price.astype(float); d['km']=d.km_driven.astype(float)
    d['fuel']=d.fuel.replace({'LPG':'CNG'}); d=d[d.fuel.isin(['Petrol','Diesel','CNG'])]
    d['trans']=(d.transmission=='Automatic').astype(int); d['indiv']=(d.seller_type=='Individual').astype(int)
    d['seller']=d.seller_type.replace({'Trustmark Dealer':'Dealer'})
    d['age']=REF_YEAR-d.year
    return d.dropna(subset=['price','km','age','engine_cc','power','seats'])
def load_bikes():
    d=pd.read_csv(f'{D}/bikes.csv').drop_duplicates()
    d['make']=d.Brand.str.strip().replace({'Royal':'Royal Enfield'}); d['model']=d.Model.str.replace(r'\s+',' ',regex=True).str.strip(); d['variant']=d.model
    d['price']=d.Selling_Price.astype(float); d['km']=d.KM_Driven.astype(float); d['year']=d.Year
    d['owner_n']=d.Owner.map(OWN); d=d.dropna(subset=['owner_n']); d['indiv']=(d.Seller_Type=='Individual').astype(int)
    d['age']=REF_YEAR-d.year; d['fuel']='Petrol'; d['trans']=0
    return d
def te_fit(train, col, y, prior, k=5):
    g=train.groupby(col)[y].agg(['sum','count']); return {i:(r['sum']+k*prior)/(r['count']+k) for i,r in g.iterrows()}
def te_oof(df,col,prior,k=5):
    out=pd.Series(index=df.index,dtype=float)
    for tr,va in KFold(5,shuffle=True,random_state=1).split(df):
        m=te_fit(df.iloc[tr],col,'ly',prior,k); out.iloc[va]=df.iloc[va][col].map(m).fillna(prior).values
    return out
def build(cat,df):
    df=df.copy(); df['ly']=np.log(df.price)
    feats_num=['age','km','owner_n','indiv']+(['trans','engine_cc','power','seats','fuel_d','fuel_c'] if cat=='car' else [])
    if cat=='car':
        df['fuel_d']=(df.fuel=='Diesel').astype(int); df['fuel_c']=(df.fuel=='CNG').astype(int)
    prior=df.ly.mean()
    # catalog of models with enough data
    cnt=df.groupby('model').size(); ok=cnt[cnt>=(5 if cat=='car' else 4)].index
    df=df[df.model.isin(ok)].copy()
    tr,te=train_test_split(df,test_size=.2,random_state=7)
    def enc(frame,fit_on,oof):
        X=pd.DataFrame(index=frame.index)
        for c,k in (('make',8),('model',4)):
            X['te_'+c]=te_oof(frame,c,prior,k) if oof else frame[c].map(te_fit(fit_on,c,'ly',prior,k)).fillna(prior)
        if cat=='car' and True:
            pass
        X['age']=frame.age; X['km']=frame.km; X['lkm']=np.log1p(frame.km); X['kmpa']=frame.km/(frame.age.clip(lower=0.5))
        for f in [x for x in feats_num if x not in('age','km')]: X[f]=frame[f]
        return X
    for d_ in (tr,te): d_['ly']=np.log(d_.price)
    Xtr=enc(tr,tr,True); Xte=enc(te,tr,False)
    P=dict(objective='quantile',learning_rate=.08,num_leaves=15,min_data_in_leaf=12,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,lambda_l2=1.0,verbose=-1,seed=3)
    models={}; preds={}
    for a in (.1,.5,.9):
        b=lgb.train({**P,'alpha':a},lgb.Dataset(Xtr,tr.ly),num_boost_round=220); preds[a]=b.predict(Xte)
    y=te.price.values; med=np.exp(preds[.5]); lo=np.exp(np.minimum(preds[.1],preds[.5])); hi=np.exp(np.maximum(preds[.9],preds[.5]))
    ly=np.log(y); lo_l=np.minimum(preds[.1],preds[.5]); hi_l=np.maximum(preds[.9],preds[.5])
    sc=np.maximum(lo_l-ly,ly-hi_l); h=len(sc)//2; q=float(max(0,np.quantile(sc[:h],min(1,0.8*(1+1/h)))))
    cov_raw_B=float(np.mean((ly[h:]>=lo_l[h:])&(ly[h:]<=hi_l[h:]))*100); cov_cal_B=float(np.mean((ly[h:]>=lo_l[h:]-q)&(ly[h:]<=hi_l[h:]+q))*100)
    metrics=dict(widen_log=q,range_coverage_raw_pct=cov_raw_B,range_coverage_calibrated_pct=cov_cal_B,coverage_eval_n=int(len(sc)-h),holdout_n=int(len(te)),train_n=int(len(tr)),median_abs_pct_error=float(np.median(np.abs(med-y)/y)*100),
        mean_abs_pct_error=float(np.mean(np.abs(med-y)/y)*100),nominal_coverage_pct=80.0,
        baseline_median_abs_pct_error=float(np.median(np.abs(np.median(tr.price)-y)/y)*100))
    # final: refit on all data with OOF encodings
    full=df; Xf=enc(full,full,True); fin={}
    parity=[]; fb={}
    for a in (.1,.5,.9):
        b=lgb.train({**P,'alpha':a},lgb.Dataset(Xf,full.ly),num_boost_round=220); fin[str(a)]=b.dump_model()['tree_info']; fb[a]=b
    for i in (0,10,200):
        parity.append(dict(x=Xf.iloc[i].astype(float).tolist(),p={str(a):float(fb[a].predict(Xf.iloc[[i]])[0]) for a in fb}))
        # sanity: node evaluator parity is checked in tests
    enc_tables={c:te_fit(full,c,'ly',prior,k) for c,k in (('make',8),('model',4))}
    # catalogs
    cat_=[]
    for (mk,md),g in full.groupby(['make','model']):
        e=dict(make=mk,model=md,n=int(len(g)),median_price=float(g.price.median()),min_year=int(g.year.min()),max_year=int(g.year.max()),median_km=float(g.km.median()))
        if cat=='car':
            e.update(engine_cc=float(g.engine_cc.median()),power=float(g.power.median()),seats=float(g.seats.median()),fuels=sorted(g.fuel.unique().tolist()),
                     auto=bool(g.trans.mean()>0), manual=bool((1-g.trans).mean()>0))
            e['by_fuel']={f:dict(engine_cc=float(gf.engine_cc.median()),power=float(gf.power.median()),seats=float(gf.seats.median()),n=int(len(gf))) for f,gf in g.groupby('fuel')}
            e['variants']=[dict(name=v,fuel=gv.fuel.iloc[0],trans=int(gv.trans.mode().iloc[0]),engine_cc=float(gv.engine_cc.median()),power=float(gv.power.median()),seats=float(gv.seats.median()),n=int(len(gv))) for v,gv in g.groupby('variant') if len(gv)>=2][:60]
        cat_.append(e)
    # aggregates (real, from data): retention by age (median price by age relative to age<=2 median), per-year km
    agg=full.groupby('age').agg(n=('price','size'),median_price=('price','median'),median_km=('km','median')).reset_index()
    agg=agg[agg.n>=8]; agg['age']=agg.age.astype(int)
    comps=full[['make','model','variant','year','km','owner_n','indiv','price']+(['fuel','trans'] if cat=='car' else [])].copy()
    comps['indiv']=comps.indiv.astype(int); comps['price']=comps.price.astype(int); comps['km']=comps.km.astype(int); comps['year']=comps.year.astype(int)
    if cat=='car': comps['trans']=comps.trans.astype(int)
    feature_names=list(Xf.columns)
    return dict(feature_names=feature_names,trees=fin,enc={k:{str(a):float(b) for a,b in v.items()} for k,v in enc_tables.items()},prior=float(prior),
        metrics=metrics,parity=parity,catalog=cat_,by_age=agg.to_dict('records'),rows=int(len(full)),
        comps=comps.to_dict('records'),median_kmpa=float((full.km/full.age.clip(lower=.5)).median()),
        year_min=int(full.year.min()),year_max=int(full.year.max()),price_min=float(full.price.min()),price_max=float(full.price.max()),
        km_max=float(full.km.max()),owners=int(full.owner_n.max()))
out={'ref_year':REF_YEAR,'categories':{'car':build('car',load_cars()),'bike':build('bike',load_bikes())}}
json.dump(out,open(f'{OUT}/model.json','w'),separators=(',',':'))
for k,v in out['categories'].items(): print(k,v['rows'],'rows',v['metrics'])
