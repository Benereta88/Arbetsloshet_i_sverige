
Slide: Histogram 

Histogrammets x-axel visar arbetslöshetsersättning utbetald i svenska kronor och y-axeln visar frekvensen av dessa belopp. Belopp är grupperade i intervall (bingar), t.ex. från 600 miljoner kronor till 1,8 miljarder kronor, t.ex. Den höga tröskeln på 700 miljoner kronor gör att många a-kassor har gjort utbetalningar av denna storlek. Den blå linjen visar sannolikhetsfördelningen med en topp på cirka 700 miljoner kronor vilket innebär att beloppet är det vanligaste och högre belopp är mer sällsynt. 

 

Slide: Scatter plot 

Från åldern 25-29 år får kvinnor högre belopp än män. Anledningarna till detta kan bero på exempelvis deltidsanställning eller föräldraledighet. I förhållande med tidigare inkomster kan kvinnor få mer betalt i ersättning.  

 

Slide: Procentuell förändring 

 

I detta diagram ser vi hur det blev tydliga svängningar där både män och kvinnor sjönk under –20% början av år 2000. Däremot ökade mäns ersättning år 2003 till 30% medan det steg långsamt för kvinnor. År 2008 där finanskrisen tog fart blev det en kraftig minskning men 2009 gick det upp till 70% för män medan det blev 19% för kvinnor. Från 2011 till 2015 var det stabilt ca 10 till 15%. Däremot började pandemin 2020 och män nådde till 70% jämfört med kvinnor som steg till 60%. Blev en ny minskning 2022 med 20% där både män och kvinnor år 2024 steg till 40% 

 

 

Slide: Stapeldiagram 

Det här stapeldiagrammet visar jämförelse av utbetalningar per arbetslöshetsförsäkringsfond och kön år 2024. Man kan se varje arbetslöshetsförsäkringsfond representeras, där blå färg står för män och röd färg för kvinnor. På vänster sida visas den totala utbetalningen i miljarder kronor, t.ex. 1,0 = 1 miljard kronor. Om man kollar på exempelvis Unionens arbetslöshetsförsäkringsfond, ser man att de betalat ut mest ersättning och att män fick lite mer än kvinnor år 2024. 

 

Slide: Regression för män 

I detta linjära regressionsdiagram är de blå punkterna de verkliga värdena för arbetslöshetsersättning till män samt det år de inträffade. Den röda linjen visar den förutsagda trenden för arbetslöshetsersättningen över tid för män. Här kan vi se att arbetslöshetsersättningen förväntas vara ca 8 miljarder SEK för år 2025. 

 

Slide: Regression för kvinnor 

I detta linjära regressionsdiagram är de blå punkterna de verkliga värdena för arbetslöshetsersättning till kvinnor samt det år de inträffade. Den röda linjen visar den förutsagda trenden för arbetslöshetsersättningen över tid för kvinnor. Här kan vi se att arbetslöshetsersättningen förväntas vara ca 5 miljarder SEK för år 2025. 

Slide: Tårtdiagrammet 

Det här cirkeldiagrammet visar fördelningen av utbetald arbetslöshetsersättning mellan män och kvinnor i procent. Det visar att av den totala utbetalningen för år 2026 kommer 53,9 % gå till män och 46,1 % till kvinnor. Diagrammet indikerar att män kommer få en större andel av ersättningen jämfört med kvinnor år 2026. Detta kan bero på att fler män kan vara arbetslösa och därför ha rätt till mer ersättning. Det kan även bero på att män i genomsnitt haft högre inkomster före arbetslöshet, vilket påverkar ersättningsnivån, eller att fler män arbetar i yrken med hög arbetslöshetsrisk, vilket påverkar statistiken. 

Slutsats 

Det vi har kommit fram till är att det är högre ersättning för män. Män och kvinnor har haft kraftiga nedgångar i ersättning vilket beror på faktorer som exempelvis kriser eller ekonomiska konjunkturer. I det procentuella diagrammet ser vi hur finanskrisen började år 2008 till 2010 där det blev en ökning i ersättning. Början av 2020 när pandemin tog fart då fler personer blev arbetslösa. Däremot minskade ersättningen för både män och kvinnor till 2018 innan den stiger upp under pandemin.  Även fast männens ersättning har varit högre så följer kvinnorna också samma likande trender 


# Arbetsloshet_i_sverige
# Antal personer som har fått arbetslöshetsersättning - per arbetslöshetskassa, kön och år


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_file_path = r""C:\Users\benereta-hoxha\Desktop\python\arbetslöshet_i_sverige\paid_unemployment_benefit_fund_year.csv"
df = pd.read_csv(csv_file_path, sep=',')


df.columns = df.columns.str.strip()

#Fyller saknade värden med 0
df.fillna(0, inplace=True)

# Summera 'amount_sek' per år och kön
df_total = df.groupby(['year', 'gender'], as_index=False).agg({'amount_sek': 'sum'})

#Beräkna procentuell förändring från föregående år
df_total['percent_change'] = df_total.groupby('gender')['amount_sek'].pct_change() * 100


output_csv_path = r"C:\Users\benereta-hoxha\Desktop\python\arbetslöshet_i_sverige\processed_unemployment_data.csv"
df_total.to_csv(output_csv_path, index=False)
print(f"Bearbetad data har sparats till {output_csv_path}")

# VISUALISERINGAR 

#1. Histogram över totalt belopp för arbetslöshetsersättning
plt.figure(figsize=(10, 6))
sns.histplot(df_total['amount_sek'], bins=20, kde=True)
plt.title('Histogram för totalt belopp av arbetslöshetsersättning')
plt.xlabel('Belopp i SEK')
plt.ylabel('Frekvens')
plt.show()

#2. Linjediagram för att visa trender över tid för varje kön
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_total, x='year', y='amount_sek', hue='gender', marker='o')
plt.title('Linjediagram för arbetslöshetsersättning över tid för varje kön')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend(title='Kön')
plt.show()

#3.Scatter plot för att visa relationen mellan olika åldersgrupper och belopp
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age_range_year', y='amount_sek', hue='gender', palette='Set2')
plt.title('Relation mellan åldersgrupper och arbetslöshetsersättning')
plt.xlabel('Åldersgrupp')
plt.ylabel('Belopp i SEK')
plt.legend(title='Kön')
plt.xticks(rotation=45)
plt.show()

#4.procentuell förändring över tid
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_total, x='year', y='percent_change', hue='gender', marker='o')
plt.axhline(0, linestyle='--', color='gray')  # Nollinje för att visa när det minskar
plt.title('Procentuell förändring av arbetslöshetsersättning över tid')
plt.xlabel('År')
plt.ylabel('Förändring i %')
plt.legend(title='Kön')
plt.show()


# 5. Scatterplot: Korrelation mellan dagar och utbetalning
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='days', y='amount_sek', hue='gender', size='year', sizes=(50, 200), palette='coolwarm', alpha=0.7)
plt.title('Korrelation mellan dagar och utbetalning', fontsize=16)
plt.xlabel('Antal dagar', fontsize=14)
plt.ylabel('Utbetalning (SEK)', fontsize=14)
plt.tight_layout()
plt.show()


#6. : Filtrera data för året 2024
df_2024 = df[df['year'] == 2024]

# Kontrollera om 'unemployment_insurance_fund' finns i datasetet
if 'unemployment_insurance_fund' not in df.columns:
    print("'unemployment_insurance_fund' kolumnen saknas i datasetet.")
else:
    # Steg 10: Gruppdata per 'unemployment_insurance_fund' och kön, samt summering av utbetalningar
    grouped_fund_gender = df_2024.groupby(['unemployment_insurance_fund', 'gender'])['amount_sek'].sum().unstack()


    # Steg 11: Visualisering av jämförelse av utbetalningar per 'unemployment_insurance_fund' och kön
    plt.figure(figsize=(14, 7))
    grouped_fund_gender.plot(kind='bar', figsize=(12, 6), cmap='coolwarm', width=0.8)

    plt.title('Jämförelse av utbetalningar per arbetslöshetsförsäkringsfond och kön (2024)', fontsize=16)
    plt.xlabel('Arbetslöshetsförsäkringsfond', fontsize=14)
    plt.ylabel('Totala utbetalningar (SEK)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Kön', title_fontsize='13')
    plt.tight_layout()
    plt.show()
    
#MASKININLÄRNING: REGRESSION

#Regression för män
df_regression_men = df_total[df_total['gender'] == 'men'] #Skapar DataFrame för män
X_men = df_regression_men[['year']] #Oberoende variabel som innehåller year , INPUT
y_men = df_regression_men['amount_sek'] #Beroende variabel som innehåller amount_sek, OUTPUT


#Dela data i träninng 80% och test 20%
X_train_men, X_test_men, y_train_men, y_test_men = train_test_split(X_men, y_men, test_size=0.2, random_state=42) #random_state=42 för att få samma resultat varje gång


#Träna modellen
model_men = LinearRegression()
model_men.fit(X_train_men, y_train_men)

#Prediktion för män
y_pred_men = model_men.predict(X_test_men)

#Förutsägelse för nästa år (2025) för män
pred_men_2025 = model_men.predict([[2026]])[0]
print(f"Förutsagd arbetslöshetsersättning för män 2025: {pred_men_2025:.2f} SEK")

#Resultatet av regressionen för män
plt.figure(figsize=(10, 6))
plt.scatter(X_test_men, y_test_men, label='Riktiga värden', color='blue')
plt.plot(X_test_men, y_pred_men, label='Prediktion', color='red')
plt.scatter(2026, pred_men_2025, color='green', label='Förutsägelse för 2025', zorder=5)
plt.title('Regression för att förutsäga arbetslöshetsersättning över tid (män)')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend()
plt.show()

#Regression för kvinnor
df_regression_women = df_total[df_total['gender'] == 'women']
X_women = df_regression_women[['year']]
y_women = df_regression_women['amount_sek']

X_train_women, X_test_women, y_train_women, y_test_women = train_test_split(X_women, y_women, test_size=0.2, random_state=42)

#Träna modellen
model_women = LinearRegression()
model_women.fit(X_train_women, y_train_women)

#Prediktion för kvinnor
y_pred_women = model_women.predict(X_test_women)

#Förutsägelse för nästa år (2025) för kvinnor
pred_women_2025 = model_women.predict([[2025]])[0]
print(f"Förutsagd arbetslöshetsersättning för kvinnor 2025: {pred_women_2025:.2f} SEK")

#Resultatet av regressionen för kvinnor
plt.figure(figsize=(10, 6))
plt.scatter(X_test_women, y_test_women, label='Riktiga värden', color='blue')
plt.plot(X_test_women, y_pred_women, label='Prediktion', color='red')
plt.scatter(2025, pred_women_2025, color='green', label='Förutsägelse för 2025', zorder=5)
plt.title('Regression för att förutsäga arbetslöshetsersättning över tid (kvinnor)')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend()
plt.show()


df_total = df.groupby('gender', as_index=False).agg({'amount_sek': 'sum'})

#Skapa regression för förutsägelse för nästa år
df_regression = df_total[['gender', 'amount_sek']]
X = np.array([0, 1]).reshape(-1, 1)  # 0 för män, 1 för kvinnor
y = df_regression['amount_sek'].values
model = LinearRegression().fit(X, y)

#Förutsägelse för 2026
pred_2026 = model.predict([[0], [1]])


#Tårtdiagram för fördelning mellan män och kvinnor
plt.figure(figsize=(6, 6))
sizes = df_total['amount_sek'].values
labels = df_total['gender']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'pink'])
plt.title('Fördelning av arbetslöshetsersättning mellan män och kvinnor')
plt.show()

print(f"Förutsägelse för män 2026: {pred_2026[0]:.2f} SEK")
print(f"Förutsägelse för kvinnor 2026: {pred_2026[1]:.2f} SEK")



    # Steg 12: Prediktion för 2025 baserat på 2024 data och antagande om 5% ökning

    # Kontrollera om 'unemployment_insurance_fund' finns i datasetet
if 'unemployment_insurance_fund' not in df.columns:
        print("'unemployment_insurance_fund' kolumnen saknas i datasetet.")
else:
        # Gruppdata per 'unemployment_insurance_fund' och kön för 2024
        grouped_fund_gender_2024 = df_2024.groupby(['unemployment_insurance_fund', 'gender'])['amount_sek'].sum().unstack()

        # Beräkna prediktion för 2025 med 5% ökning
        prediction_2025 = grouped_fund_gender_2024 * 1.05

        # Visualisering av prediktion för 2025
        plt.figure(figsize=(14, 7))
        prediction_2025.plot(kind='bar', figsize=(12, 6), cmap='coolwarm', width=0.8, color=['skyblue', 'pink'])

        plt.title('Prediktion av utbetalningar per arbetslöshetsförsäkringsfond och kön (2025)', fontsize=16)
        plt.xlabel('Arbetslöshetsförsäkringsfond', fontsize=14)
        plt.ylabel('Totala utbetalningar (SEK)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Kön', title_fontsize='13')
        plt.tight_layout()
        plt.show()

# 7. Heatmap: Korrelation mellan antal dagar och utbetalningar
corr = df[['days', 'amount_sek']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Korrelationskoefficient'})
plt.title('Korrelation mellan dagar och utbetalningar', fontsize=16)
plt.tight_layout()
plt.show()

# 4. Boxplot: Fördelning av ersättning per dag per kön
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='gender', y='amount_sek', palette='Set2')
plt.title('Fördelning av ersättning per dag per kön', fontsize=16)
plt.xlabel('Kön', fontsize=14)
plt.ylabel('Ersättning per dag (SEK)', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()





