import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_utils import load_solar_data, FEATURES

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(BASE,"model"); os.makedirs(MODEL,exist_ok=True)

def models():
    return {
      "Linear Regression":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LinearRegression())]),
      "Ridge Regression":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",Ridge(alpha=1))]),
      "Decision Tree":Pipeline([("imp",SimpleImputer(strategy="median")),("m",DecisionTreeRegressor(max_depth=10,random_state=42))]),
      "Random Forest":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=120,max_depth=14,random_state=42,n_jobs=-1))]),
      "Gradient Boosting":Pipeline([("imp",SimpleImputer(strategy="median")),("m",GradientBoostingRegressor(n_estimators=120,max_depth=3,random_state=42))]),
      "Extra Trees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=120,max_depth=14,random_state=42,n_jobs=-1))])
    }

def main():
    df=load_solar_data(BASE); X=df[FEATURES]; y=df["AC_POWER"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    all_models=models(); fitted={}; rows=[]
    for name,m in all_models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        rows.append([name,r2_score(yte,p),mean_absolute_error(yte,p),np.sqrt(mean_squared_error(yte,p))]); fitted[name]=m
    ens=Pipeline([("imp",SimpleImputer(strategy="median")),("m",VotingRegressor([
        ("rf",RandomForestRegressor(n_estimators=100,max_depth=14,random_state=42,n_jobs=-1)),
        ("gbr",GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42))]))])
    ens.fit(Xtr,ytr); p=ens.predict(Xte)
    rows.append(["Voting Ensemble",r2_score(yte,p),mean_absolute_error(yte,p),np.sqrt(mean_squared_error(yte,p))]); fitted["Voting Ensemble"]=ens
    res=pd.DataFrame(rows,columns=["Model","R2","MAE","RMSE"]).sort_values("R2",ascending=False).reset_index(drop=True)
    best=res.iloc[0]["Model"]
    joblib.dump(fitted[best],os.path.join(MODEL,"solar_model.pkl"))
    joblib.dump(FEATURES,os.path.join(MODEL,"features.pkl"))
    joblib.dump({"best_model":best,"rows":len(df),"target":"AC_POWER","r2":float(res.iloc[0].R2),"mae":float(res.iloc[0].MAE),"rmse":float(res.iloc[0].RMSE)},os.path.join(MODEL,"metadata.pkl"))
    res.to_csv(os.path.join(MODEL,"model_results.csv"),index=False)
    print(res.to_string(index=False))
    print("Saved:",best)

if __name__=="__main__": main()
