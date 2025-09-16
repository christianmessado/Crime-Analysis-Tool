import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import re

file_path = "C:/Users/beaco/Downloads/Copy of raw data.xlsx"

excel_data = pd.ExcelFile(file_path)

def safe_convert_to_float(values):
    return pd.to_numeric(values, errors='coerce').dropna().tolist()


def parse_trait_value(value):
    if isinstance(value, str):
        match = re.match(r"(\d+)(PT|NT)", value)
        if match:
            number, trait_type = match.groups()
            number = int(number)
            return number if trait_type == "PT" else -number
    return None

rape_rt = []
murder_rt = []
robbery_rt = []

traits_rape = []
traits_murder = []
traits_robbery = []

for sheet in excel_data.sheet_names:
    if sheet.isdigit():
        df = excel_data.parse(sheet)
        rape_rt.extend(safe_convert_to_float(df.get('rape.rt', [])))
        murder_rt.extend(safe_convert_to_float(df.get('murder.rt', [])))
        robbery_rt.extend(safe_convert_to_float(df.get('armed_robbery.rt', [])))

        
        traits = df.get('#of neg. and pos. traits', []).apply(parse_trait_value)
        rape_vals = pd.to_numeric(df.get('rape.rt', []), errors='coerce')
        murder_vals = pd.to_numeric(df.get('murder.rt', []), errors='coerce')
        robbery_vals = pd.to_numeric(df.get('armed_robbery.rt', []), errors='coerce')

        for t, r in zip(traits, rape_vals):
            if pd.notna(t) and pd.notna(r):
                traits_rape.append((t, r))
        for t, m in zip(traits, murder_vals):
            if pd.notna(t) and pd.notna(m):
                traits_murder.append((t, m))
        for t, ro in zip(traits, robbery_vals):
            if pd.notna(t) and pd.notna(ro):
                traits_robbery.append((t, ro))


mean_rape_rt = sum(rape_rt) / len(rape_rt) if rape_rt else float('nan')
mean_murder_rt = sum(murder_rt) / len(murder_rt) if murder_rt else float('nan')
mean_robbery_rt = sum(robbery_rt) / len(robbery_rt) if robbery_rt else float('nan')


def compute_avg_trait_response(data):
    traits, responses = zip(*data) if data else ([], [])
    return {
        "avg_traits": sum(traits)/len(traits) if traits else float('nan'),
        "avg_response": sum(responses)/len(responses) if responses else float('nan')
    }

stats_rape = compute_avg_trait_response(traits_rape)
stats_murder = compute_avg_trait_response(traits_murder)
stats_robbery = compute_avg_trait_response(traits_robbery)


def get_correlation(data):
    return pearsonr(*zip(*data)) if data else (float('nan'), float('nan'))

correlations = {
    "Rape": get_correlation(traits_rape),
    "Murder": get_correlation(traits_murder),
    "Robbery": get_correlation(traits_robbery),
}


plt.figure(figsize=(8, 6))
plt.bar(['Rape', 'Murder', 'Robbery'], 
        [mean_rape_rt, mean_murder_rt, mean_robbery_rt], 
        color=['blue', 'red', 'green'])
plt.title('Mean Response Time by Crime Type')
plt.ylabel('Response Time (rt)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


labels = ['Rape', 'Murder', 'Robbery']
avg_traits = [stats_rape["avg_traits"], stats_murder["avg_traits"], stats_robbery["avg_traits"]]
avg_rts = [stats_rape["avg_response"], stats_murder["avg_response"], stats_robbery["avg_response"]]

x = range(len(labels))
plt.figure(figsize=(10, 6))
plt.bar(x, avg_traits, width=0.4, label='Avg. Trait Score', align='center', color='purple')
plt.bar([i + 0.4 for i in x], avg_rts, width=0.4, label='Avg. Response Time (rt)', align='center', color='orange')
plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel('Value')
plt.title('Average Trait Score & Response Time by Crime Type')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


summary_df = pd.DataFrame({
    "Crime Type": labels,
    "Average Trait Score": avg_traits,
    "Average Response Time (rt)": avg_rts,
    "Correlation (r)": [round(correlations[crime][0], 3) for crime in labels],
    "p-value": [round(correlations[crime][1], 3) for crime in labels]
})

print("\n=== Trait & Response Summary Table ===")
print(summary_df.to_string(index=False))

summary_df.to_excel("trait_response_summary.xlsx", index=False)

import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd


df = pd.read_excel("trait_response_summary.xlsx")

app = dash.Dash(__name__)
app.title = "Crime Response Dashboard"

app.layout = html.Div([
    html.H1("Crime Type Summary", style={'textAlign': 'center'}),

    dcc.Graph(
        id='bar-chart',
        figure={
            'data': [
                go.Bar(name='Average Trait Score', x=df["Crime Type"], y=df["Average Trait Score"], marker_color='indigo'),
                go.Bar(name='Average Response Time', x=df["Crime Type"], y=df["Average Response Time (rt)"], marker_color='orange'),
            ],
            'layout': go.Layout(
                barmode='group',
                title='Trait Score vs Response Time by Crime Type',
                yaxis_title='Value',
                plot_bgcolor='#f9f9f9',
                paper_bgcolor='#f9f9f9'
            )
        }
    ),

    html.H3("Correlation Table", style={'textAlign': 'center'}),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([
            html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
            for i in range(len(df))
        ])
    ], style={'margin': '0 auto', 'width': '80%'})
])

if __name__ == '__main__':
    app.run(debug=True)



