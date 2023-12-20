from flask import Flask,render_template,jsonify,request
import pandas as pd
import pickle

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        
        carat= request.form.get("carat")
        
        cut=request.form.get("cut")

        color=request.form.get("color")

        clarity=request.form.get("clarity")
       
        x= request.form.get("x")
        
        y= request.form.get("y")
       
        z = request.form.get("z")
        
        print(carat,cut,color,clarity,x,y,z)
        df=pd.read_json("new.json")
        cut_encode=df["cut_encode"][df["cut"]==cut].values[0]
        color_encode=df["color_encode"][df["color"]==color].values[0]
        clarity_encode=df["clarity_encode"][df["clarity"]==clarity].values[0]
        with open('model.pkl', 'rb') as mod:
            mlmodel = pickle.load(mod)
            
        predict=mlmodel.predict([[float(carat),cut_encode,color_encode,clarity_encode,float(x),float(y),float(z)]])

        return render_template("predicted.html", predicted_value=predict[0])

    else:
        return render_template('predict.html')  


if __name__=='__main__':
    app.run(host="0.0.0.0")















