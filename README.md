# ☀️ Fini le baratin solaire  
### Un calculateur solaire basé sur des données réelles

**Baratin** est une application Streamlit permettant d’estimer de manière **physique, transparente et vérifiable** la production photovoltaïque et son interaction avec la consommation réelle d’un bâtiment.

## Fonctionnalités
- Carte interactive (Folium)
- Données météo
- Horizon réel PVGIS
- Modèle neige (encore à travailler,  mais quand même mieux que rien)
- Chargement des données CSV de smart-meter
- Modèle de la batterie.

## Calculs réalisés

    Open-Meteo  →  DNI / DHI / GHI
                        ↓
    Position solaire (azimut, élévation)
                        ↓
    Horizon PVGIS → masque DNI
                        ↓
    Projection géométrique (tilt + azimut toit)
                        ↓
    POA direct + POA diffus + albédo
                        ↓
    Neige (accumulation / fonte)
                        ↓
    Production PV réelle
                        ↓
    Comparaison avec smart-meter

## Lancer
```bash
streamlit run nsbs.py
```

## Licence
MIT
Libre pour usage personnel, éducatif et professionnel.
Merci de citer la source si utilisé 