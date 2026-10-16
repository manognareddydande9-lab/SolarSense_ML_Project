from flask import Flask,render_template,request,redirect,url_for,flash
import os,sys,subprocess,joblib,pandas as pd
from utils.data_utils import load_solar_data,FEATURES
from model_lab import MODEL_CATALOG, catalog_flat, train_supervised, run_unsupervised, LIGHTGBM_AVAILABLE

app=Flask(__name__); app.secret_key="solarsense_secret"
BASE=os.path.dirname(os.path.abspath(__file__)); MODEL=os.path.join(BASE,"model")
UPLOAD=os.path.join(BASE,"static","uploads"); os.makedirs(UPLOAD,exist_ok=True)

def meta(): return joblib.load(os.path.join(MODEL,"metadata.pkl")) if os.path.exists(os.path.join(MODEL,"metadata.pkl")) else {}
def results(): return pd.read_csv(os.path.join(MODEL,"model_results.csv")) if os.path.exists(os.path.join(MODEL,"model_results.csv")) else pd.DataFrame()

@app.route("/")
def dashboard():
    df=load_solar_data(BASE); m=meta(); r=df.tail(8).copy(); r["DATE_TIME"]=r["DATE_TIME"].dt.strftime("%d-%m-%Y %H:%M")
    return render_template("dashboard.html",rows=len(df),peak=round(df.AC_POWER.max(),2),avg=round(df.AC_POWER.mean(),2),
      daylight=int(df.is_daylight.sum()),recent=r.to_dict("records"),meta=m,
      chart={"labels":r.DATE_TIME.tolist(),"data":[round(x,2) for x in r.AC_POWER]})

@app.route("/analytics")
def analytics():
    df=load_solar_data(BASE); h=df.groupby("hour",as_index=False).AC_POWER.mean()
    c=df.select_dtypes("number").corr()["AC_POWER"].drop("AC_POWER").sort_values(ascending=False)
    return render_template("analytics.html",hourly={"labels":h.hour.tolist(),"data":[round(x,2) for x in h.AC_POWER]},
      corr={"labels":c.index.tolist(),"data":[round(x,3) for x in c.values]},
      missing=df.isna().sum().to_dict(),summary=df.describe().round(2).to_html(classes="table table-striped",border=0))

@app.route("/preprocessing",methods=["GET","POST"])
def preprocessing():
    df=load_solar_data(BASE); msg=None
    if request.method=="POST":
        if request.form["action"]=="duplicates": msg=f"Duplicate rows: {df.duplicated().sum()}"
        else: msg=f"Numeric missing values before filling: {int(df.isna().sum().sum())}"
    return render_template("preprocessing.html",df=df.head(20).to_html(classes="table table-hover",index=False),message=msg)

@app.route("/train",methods=["GET","POST"])
def train():
    if request.method=="POST":
        p=subprocess.run([sys.executable,os.path.join(BASE,"train.py")],capture_output=True,text=True,timeout=600)
        flash("Models trained successfully." if p.returncode==0 else p.stderr[-1200:],"success" if p.returncode==0 else "danger")
        return redirect(url_for("train"))
    return render_template("train.html",meta=meta(),results=results().to_dict("records"))


@app.route("/model-selection", methods=["GET", "POST"])
def model_selection():
    df = load_solar_data(BASE)
    columns = df.columns.tolist()
    selected = None; metrics = None; unsup = None; error = None
    if request.method == "POST":
        family = request.form.get("family", "Supervised")
        group = request.form.get("group", "Regression")
        model_name = request.form.get("model_name", "Linear Regression")
        target = request.form.get("target", "")
        try:
            if family == "Supervised":
                metrics, rows, features = train_supervised(df, target, model_name, group)
                selected = {"family": family, "group": group, "model": model_name, "target": target, "rows": rows, "features": features}
                flash(f"{model_name} trained successfully using target: {target}", "success")
            else:
                unsup = run_unsupervised(df, model_name)
                selected = {"family": family, "group": group, "model": model_name}
                flash(f"{model_name} completed successfully.", "success")
        except Exception as e:
            error = str(e)
    return render_template("model_selection.html", catalog=MODEL_CATALOG, columns=columns, selected=selected, metrics=metrics, unsup=unsup, error=error, lightgbm=LIGHTGBM_AVAILABLE)

@app.route("/predict",methods=["GET","POST"])
def predict():
    pred=None
    if request.method=="POST":
        path=os.path.join(MODEL,"solar_model.pkl")
        if not os.path.exists(path): flash("Train the model first.","warning"); return redirect(url_for("train"))
        vals={k:float(request.form[k]) for k in ["dc_power","daily_yield","ambient_temperature","module_temperature","irradiation","hour","minute","day","month","day_of_year"]}
        vals["is_daylight"]=int(vals["irradiation"]>5)
        X=pd.DataFrame([[vals[x] for x in FEATURES]],columns=FEATURES)
        pred=round(float(joblib.load(path).predict(X)[0]),2)
    return render_template("predict.html",prediction=pred)

@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method=="POST":
        f=request.files.get("dataset")
        if f and f.filename.endswith(".csv"): f.save(os.path.join(UPLOAD,f.filename)); flash("CSV uploaded. Replace the corresponding file in data/ and retrain if required.","success")
        else: flash("Please select a CSV file.","warning")
    return render_template("upload.html")

@app.route("/about")
def about(): return render_template("about.html")
@app.route("/help")
def help_page(): return render_template("help.html")

if __name__=="__main__": app.run(debug=True)
