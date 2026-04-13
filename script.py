import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import warnings
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

class VSAML_Golden_Scanner:
    def __init__(self, pesos, umbral=0.40):
        self.pesos = pesos
        self.tickers = list(pesos.keys())
        self.umbral = umbral
        self.modelo = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=5, random_state=42)
        self.p = {'caida': 3, 'vol_avg': 10, 'sma': 200}
        self.feature_names = ['Spread', 'Dist_SMA', 'Vol_Rel', 'Ret_3d']

    def entrenar(self):
        """Entrenamiento rápido para contextualizar el mercado actual"""
        dataset = []
        for t in self.tickers:
            df = yf.download(t, period="9y", interval="1d", progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df['Vol_SMA'] = df['Volume'].rolling(self.p['vol_avg']).mean()
            df['SMA200'] = df['Close'].rolling(self.p['sma']).mean()
            
            bull_cond = (df['Close'] < df['Close'].shift(self.p['caida'])) & \
                        (df['Vol_SMA'] < df['Vol_SMA'].shift(self.p['caida']))
            
            for i in range(200, len(df)-5):
                if bull_cond.iloc[i-1] and df['Close'].iloc[i-1] > df['SMA200'].iloc[i-1]:
                    # Features
                    spread = (df['High'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1]
                    dist_sma = (df['Close'].iloc[i-1] / df['SMA200'].iloc[i-1]) - 1
                    vol_rel = df['Volume'].iloc[i-1] / df['Vol_SMA'].iloc[i-1]
                    ret_3d = df['Close'].iloc[i-1] / df['Close'].iloc[i-4] - 1
                    # Target
                    idx, p_in = i, df['Open'].iloc[i]
                    while idx < len(df)-1:
                        if df['Close'].iloc[idx] < df['Low'].iloc[idx-3:idx].min(): break
                        idx += 1
                    win = 1 if df['Close'].iloc[idx] > p_in else 0
                    dataset.append([spread, dist_sma, vol_rel, ret_3d, win])
        
        df_train = pd.DataFrame(dataset, columns=self.feature_names + ['target'])
        self.modelo.fit(df_train[self.feature_names], df_train['target'])

    def escanear_vivo(self):
        resultados = []
        for t in self.tickers:
            df = yf.download(t, period="250d", interval="1d", progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df['Vol_SMA'] = df['Volume'].rolling(self.p['vol_avg']).mean()
            df['SMA200'] = df['Close'].rolling(self.p['sma']).mean()

            # Condición técnica VSA
            vsa_ok = (df['Close'].iloc[-1] < df['Close'].shift(self.p['caida']).iloc[-1]) and \
                     (df['Vol_SMA'].iloc[-1] < df['Vol_SMA'].shift(self.p['caida']).iloc[-1]) and \
                     (df['Close'].iloc[-1] > df['SMA200'].iloc[-1])
            
            prob = 0
            if vsa_ok:
                f = [[(df['High'].iloc[-1]-df['Low'].iloc[-1])/df['Close'].iloc[-1], 
                      (df['Close'].iloc[-1]/df['SMA200'].iloc[-1])-1,
                      df['Volume'].iloc[-1]/df['Vol_SMA'].iloc[-1],
                      df['Close'].iloc[-1]/df['Close'].iloc[-4]-1]]
                prob = self.modelo.predict_proba(pd.DataFrame(f, columns=self.feature_names))[0][1]

            # Diagnóstico visual
            if vsa_ok and prob > self.umbral:
                diag, color = f"🚀 COMPRA (Prob: {prob*100:.1f}%)", "#ccffcc"
            elif vsa_ok:
                diag, color = f"⚠️ VSA OK pero ML Filtra ({prob*100:.1f}%)", "#fff3cd"
            else:
                diag, color = "Sin señal", "#ffffff"
            
            resultados.append({'Ticker': t, 'Precio': round(float(df['Close'].iloc[-1]), 2), 'Diagnóstico': diag, 'Color': color})
        return pd.DataFrame(resultados)

def enviar_email(df):
    sender = os.environ.get("EMAIL_USER")
    receiver = os.environ.get("EMAIL_RECEIVER")
    password = os.environ.get("EMAIL_PASS")
    
    if not all([sender, receiver, password]): return print("Faltan credenciales.")

    html = f"""
    <html><body>
        <h2 style="color: #2e7d32;">📊 Reporte Estrategia Golden (ML 0.40)</h2>
        <p>Escaneo automático de activos VSA con filtro de Machine Learning.</p>
        <table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
            <tr style="background-color: #333; color: white;">
                <th>Ticker</th><th>Precio</th><th>Veredicto ML</th>
            </tr>
            {''.join([f'<tr style="background-color: {r.Color};"><td><b>{r.Ticker}</b></td><td>{r.Precio}</td><td>{r.Diagnóstico}</td></tr>' for _, r in df.iterrows()])}
        </table>
        <p><small>Basado en Sharpe 3.55 | Generado por Gemini AI System</small></p>
    </body></html>
    """
    msg = MIMEMultipart()
    msg["Subject"] = f"🚀 Alerta VSA-ML: {len(df[df['Diagnóstico'].str.contains('COMPRA')])} Oportunidades"
    msg["From"], msg["To"] = sender, receiver
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("Email enviado.")
    except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    mi_cartera = {"AVGO": 10, "IBM": 7, "TSLA": 9, "GOOGL": 10, "CCJ": 10, "MU": 8, "TSM": 12, "GE": 12, "JPM": 12, "LLY": 5}
    scanner = VSAML_Golden_Scanner(mi_cartera)
    scanner.entrenar()
    df_reporte = scanner.escanear_vivo()
    enviar_email(df_reporte)
