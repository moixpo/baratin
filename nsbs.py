# nsbs.py
# NSBS: No Solar BullShit
# Interactive streamlit app with solar-battery simulator to asses the real performances of a PV + storage system with the data of a smart-meter
#---------------------
# Moix P-O ✌️
# Albedo Engineering 2026
# MIT licence

#to run the app: streamlit run nsbs.py


import io
import requests
import pandas as pd
import numpy as np

import streamlit as st
import folium
from streamlit_folium import st_folium

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from PIL import Image


#home made import:
from solarsystem import *

from advanced_figures import *

#Cstes et divers
FIGSIZE_WIDTH=8
FIGSIZE_HEIGHT=6
WATERMARK_PICTURE='media/LogoAlbedo_90x380.png'
#WATERMARK_PICTURE='media/watermark_logo2.png'



# -----------------------------------------------------------
# Les fonctions
# -----------------------------------------------------------

def parser_smartmeter_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """
    Parse un CSV de smart-meter.
    Hypothèses :
      - Au moins une colonne date/heure (format lisible par pandas)
      - Au moins une colonne numérique de consommation
    Retourne un DataFrame indexé par datetime avec une colonne 'consommation'.
    Lève une exception si format incompatible.
    """
    try:
        df_raw = pd.read_csv(uploaded_file)
        #TODO: ici faire des cas spécifiques pour les formats des GRDs

    except Exception as e:
        raise ValueError(f"Impossible de lire le CSV : {e}")

    if df_raw.empty:
        raise ValueError("Le fichier CSV est vide.")

    # Détection d'une colonne datetime
    datetime_col = None
    for col in df_raw.columns:
        try:
            dt = pd.to_datetime(df_raw[col], errors="raise")
            # Si ça ne plante pas et qu'il n'y a pas trop de NaT, on prend
            if dt.notna().mean() > 0.9:
                datetime_col = col
                df_raw[col] = dt
                break
        except Exception:
            continue

    if datetime_col is None:
        raise ValueError(
            "Impossible de détecter une colonne de date/heure dans le CSV.\n"
            "Merci de fournir un fichier avec une colonne de timestamps."
        )

    df = df_raw.set_index(datetime_col).sort_index()

    # Détection d'une colonne de consommation (numérique)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError(
            "Aucune colonne numérique trouvée pour la consommation.\n"
            "Merci de fournir un fichier avec au moins une colonne de valeurs numériques."
        )

    # Pour l’instant, on prend la première colonne numérique
    cons_col = numeric_cols[0]
    df = df[[cons_col]].rename(columns={cons_col: "consommation"})

    # On supprime les lignes vides / NaN
    df = df.dropna(subset=["consommation"])



    if df.empty:
        raise ValueError("Aucune donnée de consommation exploitable.")

    return df


def fetch_meteo_plus(latitude, longitude, tilt, azimuth, start_date, end_date):
    """
    Récupère l'irradiance globale inclinée (GTI), l'irradiance normale et le diffu horaire pour une période donnée
    via l'API Historical Weather (Open-Meteo).
    """

    OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "global_tilted_irradiance,direct_normal_irradiance,diffuse_radiation,temperature_2m,precipitation",
        "tilt": tilt,
        "azimuth": azimuth,
        "timezone": "auto",
    }
    r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "gti_wm2": pd.to_numeric(data["hourly"]["global_tilted_irradiance"], errors="coerce"),
        "dni_wm2": pd.to_numeric(data["hourly"]["direct_normal_irradiance"], errors="coerce"),
        "dhi_wm2": pd.to_numeric(data["hourly"]["diffuse_radiation"], errors="coerce"),
        "temp_c": pd.to_numeric(data["hourly"]["temperature_2m"], errors="coerce"),
        "precip_mm": pd.to_numeric(data["hourly"]["precipitation"], errors="coerce"),
    }).dropna()

    return df


def resample_15min(df: pd.DataFrame, time_col: str, value_cols: list) -> pd.DataFrame:
    """
    Convertit une série temporelle en quarts d'heure avec interpolation linéaire.
    """
    out = df.copy()
    out = out.set_index(time_col).sort_index()
    out = out[value_cols]
    # Interpolation sur l’axe temporel après resampling
    out_15 = out.resample("15min").interpolate("time")
    out_15 = out_15.reset_index()
    out_15 = out_15.rename(columns={"index": time_col})
    return out_15


def compute_snow_model(
        df,
        temp_col="temp_c",
        precip_col="precip_mm",
        dt_hours=0.25,                    # 15 minutes
        snow_acc_factor=0.90,              # 1 mm eau -> 1 mm équiv. stock
        snow_temp_threshold=0.0,          # en-dessous => neige
        melt_base_temp=0.5,               # au-dessus de cette T°, fonte
        melt_coeff_mm_per_h_per_deg=0.75,  # mm/h/°C, à ajuster
        snow_block_threshold_mm=1.0       # seuil de blocage du panneau
        ):
    """
    Ajoute deux colonnes au DataFrame :
      - snow_load_mm : stock de neige (mm équiv. eau)
      - snow_blocked : 1 si le panneau est considéré bloqué, 0 sinon
    """
    df = df.copy()
    temps = df[temp_col].values
    precs = df[precip_col].values

    snow_storage = []
    blocked = []
    s = 0.0  # stock courant

    for T, P in zip(temps, precs):
        # 1) Accumulation : si T <= seuil neige, toute la précip est neige
        snowfall = P * snow_acc_factor if T <= snow_temp_threshold else 0.0
        s += snowfall

        # 2) Fonte : uniquement si T > melt_base_temp
        if T > melt_base_temp:
            melt_rate_mm_h = (T - melt_base_temp) * melt_coeff_mm_per_h_per_deg
            s = max(0.0, s - melt_rate_mm_h * dt_hours)

        snow_storage.append(s)
        blocked.append(1 if s > snow_block_threshold_mm else 0)

    df["snow_load_mm"] = snow_storage
    df["snow_blocked"] = blocked
    return df


def plot_snow_model(df, time_col="time"):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
    )

    # 1) Précipitations (barres) + Température (ligne)
    fig.add_trace(
        go.Bar(
            x=df[time_col],
            y=df["precip_mm"],
            name="Précipitations (mm)",
            opacity=0.6,
        ),
        row=1, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["temp_c"],
            name="Température (°C)",
            mode="lines",
        ),
        row=1, col=1, secondary_y=True,
    )

    fig.update_yaxes(
        title_text="Précipitations (mm)",
        secondary_y=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Température (°C)",
        secondary_y=True,
        row=1, col=1,
    )

    # 2) Stock de neige sur panneau
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["snow_load_mm"],
            name="Stock de neige (mm équiv. eau)",
            mode="lines",
            fill="tozeroy",
        ),
        row=2, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Stock de neige (mm)",
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Temps", row=2, col=1)
    fig.update_layout(
        title="Modèle de neige sur panneaux – précipitations, température, stock",
        legend_title_text="",
    )

    return fig


def compute_snow_model_with_irradiance(
    df,
    temp_col="temp_c",
    precip_col="precip_mm",
    poa_col="poa_global_shaded_wm2",
    dt_hours=0.25,                    # 15 minutes
    snow_acc_factor=0.95,              # 1 mm eau -> 1 mm stock neige
    snow_temp_threshold=-0.0,          # T <= seuil => neige
    melt_base_temp=0.5,               # au-dessus de cette T°, fonte air
    melt_coeff_mm_per_h_per_deg=0.5,  # fonte air : mm/h/°C
    rad_efficiency=0.2,               # fraction du flux solaire qui sert à fondre la neige
    rain_melt_factor=1.0,             # mm de neige fondue par mm de pluie "chaude"
    snow_block_threshold_mm=5.0       # seuil de blocage des panneaux
):
    """
    Modèle simple de neige sur panneaux :

    - Accumulation si T <= snow_temp_threshold : toute la précipitation est neige.
    - Fonte due à :
        * la température (degré-heure)
        * l'irradiance directe
        * la pluie au-dessus du seuil de neige (pluie "chaude")

    Hypothèse importante :
        precip_mm est le cumul de précipitation sur le pas de temps dt (ici 15 min).

    A faire: améliorer le modèle de fonte par le rayonnement (actuellement linéaire avec une efficacité fixe).
    """

    df = df.copy()
    temps = df[temp_col].values
    precs = df[precip_col].values
    poa = df[poa_col].values

    snow_storage = []
    blocked = []

    # Chaleur latente de fusion de la glace (J/kg) ~ J/mm d'eau par m²
    L_fusion = 3.34e5  # J/kg

    s = 0.0  # stock courant de neige (mm équiv. eau)

    for T, P, G in zip(temps, precs, poa):

        # 1) Accumulation de neige si T assez froid
        #    P est supposé en mm pendant le pas de temps dt_hours.
        snowfall = P * snow_acc_factor if T <= snow_temp_threshold else 0.0
        s += snowfall

        # 2) Fonte due à la température (degré-heure)
        if T > melt_base_temp:
            melt_temp_mm_h = (T - melt_base_temp) * melt_coeff_mm_per_h_per_deg
        else:
            melt_temp_mm_h = 0.0

        # 3) Fonte due au rayonnement
        if G > 0:
            # GTI moyen sur le pas de temps, approx. W/m²
            # Énergie horaire si on prolonge à 1 h :
            #   E = G * 3600 J/m²
            melt_rad_mm_h_theoretical = G * 3600.0 / L_fusion
            melt_rad_mm_h = melt_rad_mm_h_theoretical * rad_efficiency
        else:
            melt_rad_mm_h = 0.0

        # 4) Fonte due à la pluie "chaude"
        #    Si T > snow_temp_threshold, la précipitation est pluie, et contribue à la fonte.
        if P > 0 and T > snow_temp_threshold:
            # Hypothèse simple : chaque mm de pluie fait fondre rain_melt_factor mm de neige
            melt_rain_mm = rain_melt_factor * P
        else:
            melt_rain_mm = 0.0

        # 5) Fonte totale sur le pas de temps
        melt_rate_mm_h = melt_temp_mm_h + melt_rad_mm_h  # mm/h
        melt_total_mm = melt_rate_mm_h * dt_hours + melt_rain_mm  # mm sur dt

        s = max(0.0, s - melt_total_mm)

        snow_storage.append(s)
        blocked.append(1 if s > snow_block_threshold_mm else 0)

    df["snow_load_mm"] = snow_storage
    df["snow_blocked"] = blocked
    return df


def apply_snow_loss(df, pv_col="production_pv_kwh"):
    df = df.copy()
    # hypothèse simple : 0 si bloqué, 1 sinon
    df["production_pv_kwh_sans_neige"] = df[pv_col]
    df["production_pv_kwh_avec_neige"] = df[pv_col] * (1 - df["snow_blocked"])
    return df


def apply_snow_loss_two_thresholds(
    df,
    pv_col="production_pv_kwh",
    snow_load_col="snow_load_mm",
    snow_threshold_start_mm=0.5,   # début impact
    snow_threshold_full_mm=5.0     # couverture totale
):
    """
    Applique une perte de production liée à la neige avec deux seuils :

    - snow_load <= snow_threshold_start_mm  ->  aucune perte (prod = 100%)
    - snow_load >= snow_threshold_full_mm   ->  prod nulle (0%)
    - entre les deux -> perte linéaire entre 0% et 100%

    Ajoute au DataFrame :
      - snow_loss_fraction : fraction de production perdue (0 à 1)
      - production_pv_kwh_sans_neige
      - production_pv_kwh_avec_neige
    """
    df = df.copy()

    s = df[snow_load_col].astype(float)

    # Fraction de perte : 0 en dessous de t1, 1 au-dessus de t2, linéaire entre les deux
    denom = (snow_threshold_full_mm - snow_threshold_start_mm)
    if denom <= 0:
        raise ValueError("snow_threshold_full_mm doit être > snow_threshold_start_mm")

    loss_frac = (s - snow_threshold_start_mm) / denom
    loss_frac = loss_frac.clip(lower=0.0, upper=1.0)

    df["snow_loss_fraction"] = loss_frac
    df["production_pv_kwh_sans_neige"] = df[pv_col]
    df["production_pv_kwh_avec_neige"] = df[pv_col] * (1.0 - loss_frac)
    df["production_pv_kW_avec_neige"] = df["production_pv_kwh_avec_neige"].values * 4.0

    return df


def plot_pv_with_snow(df, time_col="time"):
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["production_pv_kwh_sans_neige"],
            name="PV sans neige",
            mode="lines",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["production_pv_kwh_avec_neige"],
            name="PV avec neige",
            mode="lines",
        ),
    )

    fig.update_layout(
        title="Production photovoltaïque estimée – avec et sans effet neige",
        xaxis_title="Temps",
        yaxis_title="Énergie (kWh / 15 min)",
        legend_title_text="",
    )

    return fig


def get_horizon(lat, lon):
        
    #--------------------------------
    # Solar horizon from pvgis
    #--------------------------------
    # old version:  API_HORIZON_SERIES = 'https://re.jrc.ec.europa.eu/api/printhorizon'
    API_HORIZON_SERIES = 'https://re.jrc.ec.europa.eu/api/v5_3/printhorizon'
    # API parameters
    PARAM_LATITUDE = 'lat'
    PARAM_LONGITUDE = 'lon'
    PARAM_OUTPUT_FORMAT = 'outputformat'

    OUTPUT_FORMAT='json'


    payload  = {PARAM_LATITUDE:     lat,
                                PARAM_LONGITUDE:    lon,
                                PARAM_OUTPUT_FORMAT: OUTPUT_FORMAT}

    #print("Request for horizon")

    res_horizon = requests.get(API_HORIZON_SERIES, params=payload )

    #print('Request:', res_horizon.url)
    horizon_json = res_horizon.json()

    angle_A=[]
    height_H_hor=[]

    for horizdict in horizon_json['outputs']['horizon_profile']:
        angle_A.append(horizdict['A'])
        height_H_hor.append(horizdict['H_hor'])



    angle_A_summer=[]
    height_H_summer=[]

    for horizdict in horizon_json['outputs']['summer_solstice']:
        angle_A_summer.append(horizdict['A_sun(s)'])
        height_H_summer.append(horizdict['H_sun(s)'])

    angle_A_winter=[]
    height_H_winter=[]

    for horizdict in horizon_json['outputs']['winter_solstice']:
        angle_A_winter.append(horizdict['A_sun(w)'])
        height_H_winter.append(horizdict['H_sun(w)'])
        
    horizon ={ "angle_A": angle_A,
        "height_H_hor": height_H_hor,
        "angle_A_summer": angle_A_summer,
        "height_H_summer": height_H_summer,
        "angle_A_winter": angle_A_winter,
        "height_H_winter": height_H_winter,
        }
    
    return horizon 
     

def plot_horizon(horizon):
    #--------------------------------
        
    fig_solar_horizon, axe_horizon = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))


    width = 0.35  # the width of the bars
    
    axe_horizon.fill_between(horizon["angle_A"],0,horizon["height_H_hor"], facecolor='blue', alpha=0.5)
    #axe_horizon.plot(angle_A,height_H_hor)
    axe_horizon.plot(horizon["angle_A_summer"],horizon["height_H_summer"], color='k')
    axe_horizon.plot(horizon["angle_A_winter"],horizon["height_H_winter"], color='r')
    axe_horizon.set_xlabel("Azimut [deg] E=-90,  S=0, W=90, N=180 or -180", fontsize=12)
    axe_horizon.set_ylabel("Hauteur [deg]", fontsize=12)

    axe_horizon.legend(["horizon",'hauteur du soleil été', 'hauteur du soleil hiver'])

    axe_horizon.set_title("Horizon et position du soleil", fontsize=12, weight="bold")
    axe_horizon.set_xlim(-180,180)
    axe_horizon.set_xticks(np.arange(-180, 180+1, 30.0))
    axe_horizon.grid(True)

    #addition of a watermark on the figure
    im = Image.open(WATERMARK_PICTURE)   
    fig_solar_horizon.figimage(im, 0.85*FIGSIZE_WIDTH*150, 0.15*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)



    return fig_solar_horizon


def make_horizon_interpolator(horizon):
    """
    Crée une fonction horizon(az) qui donne la hauteur d'horizon (°) pour un azimut donné (°),
    dans la convention PVGIS : 0=Sud, +Ouest, -Est, plage [-180, 180].
    """
    A = np.array(horizon["angle_A"], dtype=float)
    H = np.array(horizon["height_H_hor"], dtype=float)

    # Assure un tri croissant sur l’azimut au cas où
    order = np.argsort(A)
    A = A[order]
    H = H[order]

    def horizon_elev(az_pvgis_deg):
        """Retourne la hauteur d'horizon (°) pour un azimut ou un array d'azimuts (° PVGIS)."""
        az = np.asarray(az_pvgis_deg, dtype=float)
        # Interpolation linéaire, extrapolation plate aux bords
        return np.interp(az, A, H, left=H[0], right=H[-1])

    return horizon_elev


def get_solar_position(times, latitude, longitude):
    """
    Position solaire approx. type NOAA
    latitude : ° (Nord positif)
    longitude : ° (Est positif, Ouest négatif)
    times : DatetimeIndex (avec ou sans tz, idéalement avec tz locale)

    Retourne un DataFrame avec colonnes :
      - zenith (°)
      - elevation (°)
      - azimuth (°) 0=N, 90=E, 180=S, 270=O
    """

    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # On travaille en UTC pour l'algorithme
    times_utc = times.tz_convert("UTC") if times.tz is not None else times

    # Jour de l'année et heure fractionnaire
    doy = times_utc.dayofyear.values
    hours = times_utc.hour.values
    minutes = times_utc.minute.values
    seconds = times_utc.second.values
    frac_hour = hours + minutes / 60.0 + seconds / 3600.0

    # "Fractional year" en radians (gamma)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (frac_hour - 12.0) / 24.0)

    # Équation du temps (min)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # Déclinaison solaire (rad)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148  * np.sin(3 * gamma)
    )

    # Décalage temporel (min) en fonction de la longitude
    # (on reste en temps solaire vrai, pas en fuseau légal)
    time_offset = eqtime + 4.0 * longitude

    # Temps solaire vrai (min)
    tst = frac_hour * 60.0 + time_offset

    # Angle horaire (deg -> rad)
    ha_deg = tst / 4.0 - 180.0
    ha = np.deg2rad(ha_deg)

    lat_rad = np.deg2rad(latitude)

    # Zénith
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)
    zenith = np.rad2deg(zenith_rad)
    elevation = 90.0 - zenith

    # Pour éviter les divisions par zéro
    sin_zen = np.sin(zenith_rad)
    sin_zen[sin_zen == 0] = 1e-9

    # Azimut (formules NOAA, azimut mesuré depuis le Nord, vers l'Est)
    cos_az = (np.sin(decl) - np.sin(lat_rad) * np.cos(zenith_rad)) / (np.cos(lat_rad) * sin_zen)
    sin_az = -np.sin(ha) * np.cos(decl) / sin_zen

    az = np.rad2deg(np.arctan2(sin_az, cos_az))
    # Normalisation dans [0, 360)
    az = (az + 360.0) % 360.0

    return pd.DataFrame(
        {"zenith": zenith, "elevation": elevation, "azimuth": az},
        index=times,
    )


def add_poa_with_horizon(
    df,
    surface_tilt_deg,
    surface_azimuth_deg,
    albedo=0.2,
    col_dni="dni_wm2",
    col_dhi="dhi_wm2",
    col_elev="sun_elevation",
    col_az="sun_azimuth",
    col_sun_masked="sun_masked",
):
    """
    Ajoute les colonnes POA (plane-of-array) corrigées par l'horizon à df.

    Hypothèses :
      - df[col_dni]  : DNI (W/m²)
      - df[col_dhi]  : DHI horizontale (W/m²)
      - df[col_elev] : élévation solaire (°)
      - df[col_az]   : azimut solaire (°; 0=N, 90=E, 180=S, 270=O)
      - df[col_sun_masked] : 1 si soleil masqué par horizon, 0 sinon

    Sont ajoutées :
      - dni_shaded_wm2
      - poa_direct_wm2
      - poa_diffuse_wm2
      - poa_ground_wm2 (si 'ghi_wm2' dispo)
      - poa_global_shaded_wm2
    """
    df = df.copy()

    # 1) Masque horizon sur la DNI
    dni = df[col_dni].to_numpy(dtype=float)
    sun_masked = df[col_sun_masked].to_numpy(dtype=int)
    elev = df[col_elev].to_numpy(dtype=float)
    az_sun = df[col_az].to_numpy(dtype=float)

    # Soleil en dessous de 0° d'élévation => DNI nulle de toute façon
    above_horizon = elev > 0.0
    dni_shaded = np.where(above_horizon & (sun_masked == 0), dni, 0.0)
    df["dni_shaded_wm2"] = dni_shaded

    # 2) Angle d'incidence sur le plan PV
    beta = np.deg2rad(surface_tilt_deg)
    theta_z = np.deg2rad(90.0 - elev)  # zénith
    gamma_s = np.deg2rad(az_sun)
    gamma_p = np.deg2rad(surface_azimuth_deg)

    # cos(theta_i) = cos(θz)cos(β) + sin(θz)sin(β)cos(γs - γp)
    cos_theta_i = (
        np.cos(theta_z) * np.cos(beta)
        + np.sin(theta_z) * np.sin(beta) * np.cos(gamma_s - gamma_p)
    )
    # On ne garde que les valeurs positives (sinon la face arrière)
    cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)

    poa_direct = dni_shaded * cos_theta_i

    # 3) Diffuse inclinée (modèle isotrope simple)
    dhi = df[col_dhi].to_numpy(dtype=float)
    poa_diffuse = dhi * (1.0 + np.cos(beta)) / 2.0

    # 4) Réflexion sol (si GHI disponible)
    if "ghi_wm2" in df.columns:
        ghi = df["ghi_wm2"].to_numpy(dtype=float)
        poa_ground = ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    else:
        poa_ground = np.zeros_like(poa_direct)

    # 5) POA globale
    poa_global = poa_direct + poa_diffuse + poa_ground

    df["poa_direct_wm2"] = poa_direct
    df["poa_diffuse_wm2"] = poa_diffuse
    df["poa_ground_wm2"] = poa_ground
    df["poa_global_shaded_wm2"] = poa_global

    return df




# -----------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------

st.set_page_config(
    page_title="Fini le baratin solaire... des calculs exacts ☀️🔋 ",
    layout="wide",
    page_icon='☀️',
)


#init session state variables
if "hide_info" not in st.session_state:
    st.session_state.hide_info = True  
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

if "df_conso" not in st.session_state:
    st.session_state.df_conso = None
if "df_conso_plot" not in st.session_state:
    st.session_state.df_conso_plot = None
if "periode_txt" not in st.session_state:
    st.session_state.periode_txt = "non définie"
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


st.title("☀️ Performances du solaire (beta)")
st.subheader("Des calculs exacts sur la base des données enregistrées")

if st.session_state.hide_info == False:
        
    # Create 2 columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        Cette application estime la **production photovoltaïque** sur la base de données météorologiques
        historiques et la compare à votre **consommation** obtenue grâce aux données mesurée par votre **smartmeter** qui sont maintenant obligatoirement installés.
        Fini le baratin des vendeurs un peu trop optimistes, aujourd'hui on peut calculer exactement grâce aux données quelle sera la performance du solaire sur votre profil de consommation réel.


        L'estimation se passe en 4 étapes:

        1) Entrez sur votre droite les données de base du toit, la quantité de solaire installée et la batterie, ainsi qu'une première idée des prix, par exemple avec les chiffres trouvés sur une offre.

        Puis ci-dessous:

        2) Avoir un fichier csv de votre consommation, de préférence une année entière pour avoir toutes les saisons. 
        Si le fichier ne peut pas être lu, c'est que le format n'est pas encore connu. Envoyez un email à info@autoconsommation.ch avec le fichier et le nom de votre distributeur et j'adapterai pour rendre compatible.
        3) Cliquer sur votre position sur la carte, cela est nécessaires pour obtenir l'estimation solaire.
        4) Lancer le collecte des données météorologiques.
        5) Lancer les calculs d'autoconsommation et d'autonomie --> regarder les résultats



        Notes: 
        - Aucune donnée n'est conservée après les calculs.
        - Si aucune données smartmeter n'est disponible, la simuation solaire est quand même disponible et par défaut les données de l'année 2024 sont chargées
        - Les diverses subventions ne sont pas prise en compte pour rester simple. Attention à bien identifier les offres où elles sont désuites ou non, certain n'hésite pas sur le subterfuge pour rendre leur proposition attractive comparé à d'autres.
        """
        )


    # Display the image in the center column
    with col2:
        st.image("media/CalculateurBaratin.jpg", caption="Fini l'enfumage...", width=500)

        bouton_OK_info =st.button("OK, masquer les infos", key="bouton_OK_info")
        
        if bouton_OK_info:
            st.session_state.hide_info = True
            st.rerun()


else:
    bouton_aff_info =st.button("Afficher les infos", key="bouton_aff_info")
        
    if bouton_aff_info:
        st.session_state.hide_info = False
        st.rerun()

# -----------------------------------------------------------
# Sidebar – Paramètres système PV et batterie
# -----------------------------------------------------------

st.sidebar.title("⚙️ 1- Simulation")

st.sidebar.markdown("---")
st.sidebar.header("☀️ Paramètres photovoltaïques")

pv_kw = st.sidebar.number_input(
    "Puissance photovoltaïque prévue (kWc)",
    min_value=0.1,
    max_value=30.0, 
    value=12.0,
    step=0.1,
)

orientation_deg = st.sidebar.number_input(
    "Orientation du toit ( 0°=S, -90°=E, 90°=O, ±180°=N)",
    min_value=-180.0,
    max_value=180.0,
    value=0.0,   # par défaut plein Sud
    step=1.0,
)

pente_deg = st.sidebar.number_input(
    "Pente du toit (0°=plat, 90°=vertical)",
    min_value=0.0,
    max_value=90.0,
    value=20.0,  # par défaut 20°
    step=1.0,
)


PV_total_cost_usr_input = st.sidebar.slider("Prix du PV   (CHF): ", 
                                               min_value=0.0, 
                                               max_value=50000.0, 
                                               value=15000.0, 
                                               step=100.0,
                                               help=(
                                                    "Tout inclure, sauf les batteries qui sont entrées ci-dessous \n"
                                                    )
                                                )

st.sidebar.markdown("---")
st.sidebar.header("🔋 Paramètres stockage")


st.sidebar.write("Taille totale de la batterie")



battery_size_kwh_usr_input = st.sidebar.number_input("Capacité batterie  (kWh): ", 
                                               min_value=0.0, 
                                               max_value=50.0, 
                                               value=10.0, 
                                               step=1.0,
                                               help=(
                                                    "Pour ne pas avoir de batterie, simplement mettre à 0 kWh.\n\n"
                                                    "La capacité utile sera de 80% de la capacité présentée. \n"
                                                    )
                                                )

batt_total_cost_usr_input = st.sidebar.slider("Prix de la batterie  (CHF): ", 
                                               min_value=0.0, 
                                               max_value=25000.0, 
                                               value=8000.0, 
                                               step=100.0,
                                               help=(
                                                    "Si cela n'est pas indiqué dans l'offre, mettre une partie pour que le total corresponde.\n\n"
                                                    "Un prix de 800CHF / kWh est standard. \n"
                                                    )
                                                )

st.sidebar.markdown("---")
st.sidebar.header("⚡💸 Paramètres prix")

fixed_price_buy_usr_input = st.sidebar.slider("Prix achat électricité (ct/kWh): ", min_value=5.0, max_value=40.0, value=22.1, step=0.1) / 100  # directly in CHF/kWh
fixed_price_sell_usr_input = st.sidebar.slider("Prix revente surplus PV (ct/kWh): ", min_value=5.0, max_value=40.0, value=8.5, step=0.1) / 100  # directly in CHF/kWh


# On garde en state la dernière valeur courante

# st.sidebar.markdown("---")
# st.sidebar.subheader("Point sélectionné")
# if "selected_point" not in st.session_state:
#     st.session_state.selected_point = None

# if st.session_state.selected_point is None:
#     st.sidebar.write("Point sélectionné : _aucun_")
# else:
#     st.sidebar.write(
#         f"Point sélectionné : **{st.session_state.selected_point['lat']:.5f}, "
#         f"{st.session_state.selected_point['lon']:.5f}**"
#     )

st.sidebar.markdown("---")
st.sidebar.write("AUTOCONSOMMATION.CH")
st.sidebar.write("Fini le baratin, aujourd'hui on peut calculer exactement grâce aux données des smart-meters et à l'historique météo \n")
st.sidebar.write("Version 0.3, Moix P-O, 2026, ✌️")




# -----------------------------------------------------------
# CSV du smartmeter

st.markdown("---")
st.subheader("2 - Données de consommation (smart-meter)")


col_left, col_right = st.columns([0.5, 1.5], gap="large")

with col_left:

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV du compteur intelligent",
        type=["csv"],
        key="smartmeter_uploader",  
        help=(
            "Idéalement une année complète de données pour représenter toutes les conditions.\n\n"
            "Le fichier doit contenir au moins :\n"
            "- une colonne de date/heure\n"
            "- une colonne numérique de consommation"
        ),
    )

    if uploaded_file is not None:

        # ne re-parser que si le fichier a changé
        if st.session_state.uploaded_file_name != uploaded_file.name:
            try:
                df_conso = parser_smartmeter_csv(uploaded_file)
                df_conso_plot = df_conso.reset_index().rename(columns={df_conso.index.name: "time"})

                st.session_state.df_conso = df_conso
                st.session_state.df_conso_plot = df_conso_plot

                date_debut = df_conso.index.min()
                date_fin = df_conso.index.max()
                periode_txt = f"{date_debut.date().isoformat()} → {date_fin.date().isoformat()}"
                
                st.session_state.periode_txt = periode_txt
                st.session_state.uploaded_file_name = uploaded_file.name
                
                st.success(f"Données de consommation chargées.\nPériode : {periode_txt} ")
            except Exception as e:
                st.error(
                    "Le fichier fourni ne semble pas compatible avec le format attendu.\n\n"
                    f"Détails : {e}"
                )
                df_conso = None
                df_conso_plot = None
                st.session_state.df_conso = None
                st.session_state.df_conso_plot = None
                st.session_state.periode_txt = "non définie"
                st.session_state.uploaded_file_name = None

    #st.markdown(f"**Période d'étude** : {periode_txt}")

    #récupération des données en session state pour les cas où le fichier n'est pas rechargé
    df_conso =  st.session_state.df_conso
    df_conso_plot = st.session_state.df_conso_plot


with col_right:

    #st.markdown("### Profil de consommation (smart-meter)")

    if st.session_state.df_conso_plot is not None:
        #affiche un plotly de la conso à droite
        fig_conso = px.line(
            df_conso_plot,
            x="time",
            y="consommation",
            title="Consommation mesurée (telle que fournie)",
        )
        st.plotly_chart(fig_conso, width='stretch')
        st.success("Veuillez véfifier que la courbe de consommation semble cohérente (tous les formats de fichiers des smartmeters ne sont pas validés).")

    else:
        st.info(
            "Importez un fichier CSV de smart-meter pour afficher la "
            "courbe de consommation."
        )


if st.session_state.df_conso_plot is not None:
    timestep=0.25
    st.write("📋 Résumé des données chargées pour cette période")
    consumption_kWh = df_conso_plot["consommation"].sum() * timestep

    scaled_production_kWh = 0.0
    bill_without_solar = 0.0
    peak_power_of_consumption= 0.0
    peak_power_of_consumption = df_conso_plot["consommation"].max()
    mean_power_of_consumption = df_conso_plot["consommation"].mean()

    col1, col2, col3= st.columns(3)
    col1.metric("Consumption", str(int(consumption_kWh))+" kWh")
    col2.metric("Consumption peak", f"{peak_power_of_consumption :.1f}" + " kW")
    col3.metric("Mean Power", f"{mean_power_of_consumption :.1f}" + "kW")


    #col2.metric("Production", str(int(scaled_production_kWh))+" kWh")
    #col3.metric("Bill", f"{bill_without_solar :.0f}" + "CHF")



# -----------------------------------------------------------
# Carte Folium pour choisir un point
# -----------------------------------------------------------
st.markdown("---")
st.markdown("### 3 - Choisissez l'emplacement sur la carte")
st.write("Et vérifiez la puissance PV prévue, l'orientation et l'angle du toit dans ci-contre")
default_location = [46.23647, 7.36697]  # Tourbillon 
center = default_location
if st.session_state.selected_point is not None:
    center = [st.session_state.selected_point["lat"], st.session_state.selected_point["lon"]]

m = folium.Map(location=center, zoom_start=14 if st.session_state.selected_point else 8, control_scale=True, tiles='CartoDB Voyager')

#m = folium.Map(location=default_location, zoom_start=8, control_scale=True, tiles='CartoDB Voyager') #"CartoDB Positron"


# Marqueur actuel si déjà sélectionné
if st.session_state.selected_point is not None:
    folium.Marker(
        location=[st.session_state.selected_point["lat"], st.session_state.selected_point["lon"]],
        popup="Point sélectionné",
        tooltip="Point sélectionné",
    ).add_to(m)

map_state = st_folium(m, height=450, width=None, key="map_select")

#st.write("DEBUG: reached map section")


# Gestion du clic
if map_state and map_state.get("last_clicked"):
    lat = float(map_state["last_clicked"]["lat"])
    lon = float(map_state["last_clicked"]["lng"])

    new_point = {"lat": lat, "lon": lon}
    if st.session_state.selected_point != new_point:
        st.session_state.selected_point = new_point
        #st.rerun() ne pas faire de rerun ici, stremlit le fait automatiquement



#écrire dessous la carte les coordonnées du point sélectionné
if st.session_state.selected_point is None:
    st.write("Point sélectionné : _aucun_")
else:
    st.write(
        f"Point sélectionné : **{st.session_state.selected_point['lat']:.5f}, "
        f"{st.session_state.selected_point['lon']:.5f}**"
        )

# -----------------------------------------------------------
# Zone centrale : graphiques conso + bouton meteo
# -----------------------------------------------------------

st.markdown("---")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    
    st.markdown("### 4 - Lancer le calcul d'irradiance")
    st.caption(
        "Les données historique d'irradiance sont obtenues dans des bases de données météorologique publiques en ligne. Un modèle du relief et un modèle de la neige accumulée ont été ajoutés car c'est un des points critiques en Suisse et on observe vite une divergence si cela n'est pas pris en compte. Les dates sont ajustées à la période de consommation si elle est fournie."
    )

    bouton_calcul = st.button(
        "Récupérer l'irradiance et calculer la production PV",
        type="primary",
        disabled=(st.session_state.selected_point is None),
    )

with col_right:


    if bouton_calcul:
        if st.session_state.selected_point is None:
            st.error("Veuillez d'abord sélectionner un point sur la carte.")
        
        else: 
            lat = st.session_state.selected_point["lat"]
            lon = st.session_state.selected_point["lon"]
            horizon = get_horizon(lat, lon)
            fig_horizon = plot_horizon(horizon)
            st.pyplot(fig_horizon)


# -----------------------------------------------------------
# Calcul production PV & affichage conso + prod
# -----------------------------------------------------------
if bouton_calcul:
    if st.session_state.selected_point is None:
        st.error("Veuillez d'abord sélectionner un point sur la carte.")
    else:
        # Définition de la période pour Open-Meteo
        if df_conso is not None:
            start_date = df_conso.index.min().date().isoformat()
            end_date = df_conso.index.max().date().isoformat()
        else:
            # Par défaut, année 2024
            start_date = "2024-01-01"
            end_date = "2024-12-31"

        lat = st.session_state.selected_point["lat"]
        lon = st.session_state.selected_point["lon"]

        try:
            with st.spinner("Requête vers base de donnée et traitement des données..."):
                df_gti_hourly = fetch_meteo_plus(
                    latitude=lat,
                    longitude=lon,
                    tilt=pente_deg,
                    azimuth=orientation_deg,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Conversion GTI en pas de 15 minutes
                df_gti_15 = resample_15min(
                    df_gti_hourly,
                    time_col="time",
                    value_cols=["gti_wm2", "dni_wm2", "dhi_wm2", "temp_c","precip_mm"],
                )


                # Approximation : énergie surfacique par pas de 15 min (kWh/m²)
                # GTI est en W/m² moyenne sur l'heure précédente.
                # Sur 15 min ( soit 0.25 h) → E ≈ P * 0.25 h 
                df_gti_15["gti_kwh_m2_15min"] = df_gti_15["gti_wm2"] / 1000.0 * 0.25 

                #idem pour les précoipitations, il faut diviser par 4 par c'était des mm par heure:
                df_gti_15["precip_mm"] = df_gti_15["precip_mm"] * 0.25 


                # Estimation simplifiée de la production PV (kWh par 15 min)
                # production = kWc * kWh/m² * PR 

                # Hypothèse simple de performance ratio
                PR = 0.87 #TODO: faire évoluer avec un deuxième couche de modèle meteo, avec le ratio kWh/kWc des cartes moyennes d'ensoleillement


                df_gti_15["production_pv_kwh"] = pv_kw * df_gti_15["gti_kwh_m2_15min"] * PR 
                df_gti_15["production_pv_kW"] = df_gti_15["production_pv_kwh"] * 4

                with col_left:
                    st.success(
                        f"Données météo chargées pour la plage du {start_date} au {end_date}."
                    )


                df_sun_pos = get_solar_position(df_gti_15["time"], lat, lon)
                #corriger l'azimuth pour matcher la convention -180 à 180° aulieu de 0 à 360, ici  0 est le sud, c'est le même
                np_az = df_sun_pos["azimuth"].values - 180.0
                #np_az[np_az > 180.0 ] = np_az[np_az > 180.0 ]-360.0
                
                #et corrige l'azimth quand on pass sous l'horizon, pour ne pas voir des sauts inutiles lors des vérifications
                np_elev = df_sun_pos["elevation"].values
                np_az[np_elev < 0.0] = 0.0
                np_elev[np_elev < 0.0] = 0.0

                df_sun_pos["sun_azimuth_corrected"] =  np_az  


                df_gti_15["sun_elevation"]= np_elev
                df_gti_15["sun_azimuth"]= np_az


                #cherche la hauteur de l'horizon pour chaque moment (à l'azimuth du soleil)
                horizon_func = make_horizon_interpolator(horizon)

                H_hor = horizon_func(df_gti_15["sun_azimuth"].values)   # hauteur d'horizon à tous les azimuts de la période
                
                # #Blockage du soleil direct: à faire TODO
                sun_blocked = df_gti_15["sun_elevation"].values < H_hor        # booléen array

                # masque 0/1
                mask_direct = (sun_blocked).astype(float)
                df_gti_15["horizon_elevation"] = H_hor
                df_gti_15["sun_masked"] = mask_direct


                #calcul de l'irradiance globale sur le plan incliné, TODO: ajuster l'albédo selon les saisons et la neige
                df_poa = add_poa_with_horizon(
                    df_gti_15,
                    surface_tilt_deg=pente_deg,
                    surface_azimuth_deg=orientation_deg,
                    albedo=0.2,  # ajustable
                )

                dt_hours = 0.25  # 15 min
                df_poa["poa_kwh_m2_15min"] = df_poa["poa_global_shaded_wm2"] / 1000.0 * dt_hours

                df_poa["production_pv_kwh"] = pv_kw * df_poa["poa_kwh_m2_15min"] * PR
                df_poa["production_pv_kW"] = df_poa["production_pv_kwh"] / dt_hours

                #st.dataframe(df_sun_pos.head(48), width='stretch')
                #st.dataframe(df_gti_15.head(48), width='stretch')
                #st.dataframe(df_poa.head(48), width='stretch')

                # fig_temp_rain = make_subplots(specs=[[{"secondary_y": True}]])

                # # Précipitations (souvent cumul horaire en mm)
                # fig_temp_rain.add_trace(
                #     go.Bar(
                #         x=df_gti_15["time"],
                #         y=df_gti_15["precip_mm"],
                #         name="Précipitations (mm)"
                #     ),
                #     secondary_y=False,
                # )

                # # Température (°C)
                # fig_temp_rain.add_trace(
                #     go.Scatter(
                #         x=df_gti_15["time"],
                #         y=df_gti_15["temp_c"],
                #         name="Température (°C)",
                #         mode="lines"
                #     ),
                #     secondary_y=True,
                # )

                # # Titres axes
                # fig_temp_rain.update_yaxes(
                #     title_text="Précipitations (mm/h)",
                #     secondary_y=False
                # )

                # fig_temp_rain.update_yaxes(
                #     title_text="Température (°C)",
                #     secondary_y=True
                # )

                # fig_temp_rain.update_xaxes(title_text="Temps")
                # fig_temp_rain.update_layout(
                #     title="Température et précipitations – échelles séparées",
                #     legend_title_text=""
                # )

                # st.plotly_chart(fig_temp_rain, width='stretch')

                #st.markdown("### Modèle d'épaisseur neige")



                df_snow = compute_snow_model_with_irradiance(df_poa)
                
                #df_snow = apply_snow_loss(df_snow)
                df_snow = apply_snow_loss_two_thresholds(
                    df_snow,
                    pv_col="production_pv_kwh",
                    snow_load_col="snow_load_mm",
                    snow_threshold_start_mm=1.0,  # à ajuster
                    snow_threshold_full_mm=20.0    # à ajuster
                )




                scaled_production_kWh = df_snow["production_pv_kwh_avec_neige"].sum()
                peak_power_of_production = df_snow["production_pv_kwh_avec_neige"].max()*4.0
                col1, col2= st.columns(2)
                
                col1.metric("Production", str(int(scaled_production_kWh))+" kWh")
                #col2.metric("Production peak", f"{peak_power_of_production :.1f}" + " kW")
                col2.metric("Performance", f"{scaled_production_kWh/pv_kw :.0f}" + " kWh/kWc")

                #col3.metric("hours of production", f"{mean_power_of_consumption :.1f}" + "h")




                with st.expander("Aperçu des données météorologiques utilisées et du solaire estimé"):
                #     st.dataframe(df_snow.head(96), width='stretch')
                    fig_irrad = px.area(
                        df_gti_15,
                        x="time",
                        y=["dhi_wm2", "dni_wm2"],
                        labels={"dhi_wm2": "Diffu",
                                "dni_wm2": "Direct normal"},
                        title="Irradiance directe normale et diffuse",
                    )
                    st.plotly_chart(fig_irrad, width='stretch')


                    fig_sun_pos = px.line(
                        df_sun_pos,
                        x=df_sun_pos.index,
                        y=["elevation","azimuth"],
                        labels={"elevation": "hauteur du soleil"},
                        title="Position du soleil dans le ciel",
                    )
                    st.plotly_chart(fig_sun_pos, width='stretch')


                    #Ensuite il faut appliquer le masque de l'horizon
                    fig_hor_mask = plot_horizon(horizon)
                    #reprend les axes: 
                    axes_fig_hor_mask = fig_hor_mask.axes[0]
                    #ajoute la hauteur du soleil:

                    axes_fig_hor_mask.plot(df_gti_15["sun_azimuth"].values,
                                    df_gti_15["sun_elevation"].values,
                                    marker='+',
                                    alpha=0.15,
                                    color='b',
                                    linestyle='None')




                    axes_fig_hor_mask.plot(df_gti_15["sun_azimuth"].values,
                                    H_hor,
                                    marker='.',
                                    alpha=0.25,
                                    color='m',
                                    linestyle='None')

                    st.pyplot(fig_hor_mask)





                    fig_poa = px.line(
                        df_poa,
                        x="time",
                        y=["gti_wm2","poa_global_shaded_wm2"],
                        labels={"y": "Production PV estimée (W/m2)"},
                        title="Irradiance inclinée (GTI) et irradiance total avec masque horizon et albédo",
                    )
                    st.plotly_chart(fig_poa, width='stretch')


                    #st.markdown("### 4 - Production PV estimée (pas 15 min)")
                    fig_pv = px.line(
                        df_poa,
                        x="time",
                        y="production_pv_kW",
                        labels={"production_pv_kW": "Production PV estimée (kW)"},
                        title="Production PV estimée à partir de l'irradiance totale, incluant le masque",
                    )
                    st.plotly_chart(fig_pv, width='stretch')
                #st.markdown("### Impact de la neige sur la production PV")

                with st.expander("Aperçu du calcul de la neige"):
                    fig_snow = plot_snow_model(df_snow)
                    st.plotly_chart(fig_snow, width='stretch')

                    fig_pv_snow = plot_pv_with_snow(df_snow)
                    st.plotly_chart(fig_pv_snow, width='stretch')


                # with st.expander("Aperçu des données PV (premières lignes)"):
                #     st.dataframe(df_snow.head(96), width='stretch')

                # Si on a de la conso, on resample aussi en 15 min et on fusionne
                if df_conso is not None:

                    #Ici on peut faire des calculs d'autoconso
                    #on commence par merger les deux df dans df_pow_profile (comme dans l'app battery sizer)

                    df_conso_15 = df_conso.copy()
                    df_conso_15 = df_conso_15.resample("15min").interpolate("time")
                    df_conso_15 = df_conso_15.reset_index().rename(columns={df_conso_15.index.name: "time"})

                    # Jointure interne sur le temps
                    df_pow_profile = pd.merge_asof(
                        df_conso_15.sort_values("time"),
                        df_snow.sort_values("time"),
                        on="time",
                        direction="nearest",
                        tolerance=pd.Timedelta("10min"),
                    )

                    df_pow_profile = df_pow_profile.dropna(subset=["gti_wm2", "production_pv_kwh", "production_pv_kW"])

                    #renomme les colonnes comme dans l'app battery sizer pour pouvoir utiliser              
                    df_pow_profile.rename(columns={"consommation": "Consumption [kW]", 
                                           "production_pv_kW_avec_neige": "Solar power scaled"},
                                            inplace=True,)




                    st.write('\n \n')
                    st.markdown("---")

                    # Graphique conso + production
                    st.write('\n \n')
                    st.markdown("### 5 - Résultats ")
                    st.markdown("##### Calculs du stockage, des échanges avec le réseau, de l'autoconsommation et de l'autonomie ")

                    # st.success(
                    #     f"Données meteo du {start_date} au {end_date} "
                    #     f"et fusionnées avec le profil de consommation."
                    #     )
                    df_pow_profile = df_pow_profile.set_index("time")

                    #st.dataframe(df_pow_profile.head(48), width='stretch')

                    #TODO: provisoirement ici
                    #save hours sampling for heatmap display:
                    hours_mean_df = df_pow_profile.resample('h', label="right", closed="right").mean() 
                    day_kwh_df = hours_mean_df.resample('d').sum() 
                    month_kwh_df = day_kwh_df.resample('ME').sum() 

                    # print(day_kwh_df.head())
                    # all_channels_labels=list(day_kwh_df.columns)
                    # print(day_kwh_df)

                    #The original data set
                    # # Combined Solar Power and Energy Consumption Plot using Plotly
                    # fig_combined = px.line(df_pow_profile, x=df_pow_profile.index, 
                    #                         y=df_pow_profile["Solar power scaled", "Consumption [kW]"], 
                    #                         title="🌞 Solar Production vs ⚡ Energy Consumption", 
                    #                         labels={"value": "Power [kW]", "variable": "Legend"},
                    #                         color_discrete_sequence=["orange", "lightblue"] )
                    
                    # # Move legend below the graph
                    # fig_combined.update_layout(
                    #     legend=dict(
                    #         orientation="h",
                    #         yanchor="top",
                    #         y=-0.2,  # Position below the graph
                    #         xanchor="center",
                    #         x=0.1
                    #     )
                    # )
                    # st.plotly_chart(fig_combined)



        except Exception as e:
            st.error(f"Échec de la récupération ou du traitement des données : {e}")
            st.exception(e) #A commenter




            
#if df_conso is not None:
if "df_pow_profile" in locals() and not df_pow_profile.empty:    
    
    #*********************
    # Perform the simulation
    #*********************

    #take the data of the power profile in numpy (format for simulation)
    pow_array_all = df_pow_profile["Consumption [kW]"].to_numpy()
    solar_array_all_scaled =  df_pow_profile["Solar power scaled"].to_numpy()
    
    


    # #*********************
    # # Computations for the peak shaving and curtailment from the peak powers
    # peak_power_of_consumption = df_pow_profile["Consumption [kW]"].max()
    # clipping_level = peak_shaving_user_input*peak_power_of_consumption/100.0

    peak_power_of_production = df_pow_profile["Solar power scaled"].max()
    pv_injection_curtailment_user_input = 100.0
    pv_injection_curtailment_power = pv_injection_curtailment_user_input *peak_power_of_production/100

    #***********************
    #Load the wanted prices
    #***********************

    length_profile = len(df_pow_profile.index)

    #PV excess selling is same for both cases:
    price_array_sell_pv = np.ones(length_profile) * fixed_price_sell_usr_input
    price_array_buy = np.ones(length_profile) * fixed_price_buy_usr_input
    #and put the prices in the dataframe:
    df_pow_profile["price buy"] = price_array_buy
    df_pow_profile["price sell PV"] = price_array_sell_pv




    ##########################################
    #Let's simulate the solar system 
    #########################################################################
    # with the solarsystem.py object:

    #initialisation qui ne sont pas données en option:
    soc_init_user_input = 50.0
    batt_soc_for_backup_user_input = 20.0
    batt_soc_for_peak_user_input = 20.0
    opt_to_use_peak_shaving = False
    batt_charge_power_rate_user_input = 0.5 
    battery_charge_power_kw = batt_charge_power_rate_user_input * battery_size_kwh_usr_input
    INVERTER_STANDBY_W = 50.0
    EFFICIENCY_BATT_ONE_WAY = 0.95

    solar_system = SolarSystem("M. Vendeur","Rue du barratin 6, 2050 Transition" )

    # properties initialisation  
    solar_system.batt_capacity_kWh = battery_size_kwh_usr_input #10*1 # in kWh
    solar_system.soc_init = soc_init_user_input # in %
    solar_system.soc_for_backup_user = batt_soc_for_backup_user_input
    solar_system.soc_for_peak_shaving_user = batt_soc_for_peak_user_input
    solar_system.peak_shaving_activated = opt_to_use_peak_shaving
    solar_system.max_grid_injection_power = peak_power_of_production  # all the solar #pv_injection_curtailment_power #kW  a high value in order not to have caping 

    solar_system.peak_shaving_limit = max(peak_power_of_production,peak_power_of_consumption) #all the load or solar #clipping_level #kW

    solar_system.max_power_charge = battery_charge_power_kw #to update the max charge used by default independently of the battery size
    solar_system.max_power_discharge = -battery_charge_power_kw #same rate applied
    solar_system.max_inverter_power = 500 #kW  a high value in order not to have caping   # 15kW  for the next3

    if battery_size_kwh_usr_input == 0.0:
        solar_system.selfpowerconsumption = 0.0
    else:
        solar_system.selfpowerconsumption = INVERTER_STANDBY_W / 1000
    solar_system.efficiency_batt_one_way = EFFICIENCY_BATT_ONE_WAY

    # solar_system.gps_location = [46.208, 7.394] 
    # solar_system.pv_kW_installed = 9.24 #power installed on the roof
    # solar_system.roof_orientation = -10 # 0=S, 90°=W, -90°=E, -180°=N (or -180)
    # solar_system.roof_slope = 25.0
    # solar_system.comment = "installed in June 2022"


    #load data in the module for simulation:
    solar_system.load_data_for_simulation(pow_array_all, solar_array_all_scaled, timestep=0.25)


    ##########################################
    #Let's simulate the solar system without battery for reference:
    solar_system.run_simple_simulation(print_res=False)


    #Take the results of the simple simulation:
    grid_array_all = solar_system.net_grid_balance_profile
    reference_curtailment_lost_energy_kwh = sum(solar_system.lostproduction_profile) * timestep

    #All the series for the case with solar only and no storage will be the reference:
    df_pow_profile["grid power reference"] = grid_array_all #all grid balance without storage with scaled solar

    # Replace all positive values with 0 to have the injection only, note the injection is negative power on the grid
    df_pow_profile["grid injection reference"] = df_pow_profile["grid power reference"].mask(df_pow_profile["grid power reference"] > 0, 0.0)
    # Replace all negative values with 0 to have the consumption only
    df_pow_profile["grid consumption reference"] = df_pow_profile["grid power reference"].mask(df_pow_profile["grid power reference"] < 0, 0.0)


    consumption_kWh = df_pow_profile["Consumption [kW]"].sum() * timestep
    #original_production_kWh = df_pow_profile["Solar power [kW]"].sum() * timestep #from dataset
    scaled_production_kWh = df_pow_profile["Solar power scaled"].sum() * timestep #scaled and used

    reference_grid_injection_kWh = -df_pow_profile["grid injection reference"].sum() * timestep
    reference_grid_consumption_kWh = df_pow_profile["grid consumption reference"].sum() * timestep

    # reference_self_consumption_ratio = (scaled_production_kWh - reference_grid_injection_kWh - reference_curtailment_lost_energy_kwh) / scaled_production_kWh * 100
    # reference_autarky_ratio = (consumption_kWh - reference_grid_consumption_kWh) / consumption_kWh * 100.0  
    # print(" Check: " , reference_autarky_ratio, solar_system.autarky_rate)
    reference_self_consumption_ratio= solar_system.selfconsumption_rate
    reference_autarky_ratio = solar_system.autarky_rate


    # compute the costs based on the selected price for the consumption only, it must be done for every quarters because 
    df_pow_profile["CostForBuyingNoSolar"] = (df_pow_profile["Consumption [kW]"] * df_pow_profile["price buy"] * timestep)   
    cost_buying_no_solar_chf = df_pow_profile["CostForBuyingNoSolar"].sum()
    #print("Cost buying paid without solar and without storage:", cost_buying_no_solar_chf)

    df_pow_profile["CostForBuyingSolarOnly"] = (df_pow_profile["grid consumption reference"] * df_pow_profile["price buy"] * timestep)   
    cost_buying_solar_only_chf = df_pow_profile["CostForBuyingSolarOnly"].sum()
    #print("Cost buying paid witht solar only, no storage:", cost_buying_solar_only_chf)

    df_pow_profile["SellSolarOnly"] = -(df_pow_profile["grid injection reference"] * df_pow_profile["price sell PV"] * timestep)   
    sellings_solar_only_chf = df_pow_profile["SellSolarOnly"].sum()
    #print("Sold PV electricity with with solar only, no storage:", sellings_solar_only_chf)



    full_plim_array = np.ones(length_profile)
    full_day_max_charging_power_profile_array = np.ones(length_profile) * solar_system.max_power_charge

    #and update the max charge profile for simulation:
    solar_system.battery_max_charge_setpoint_profile = full_day_max_charging_power_profile_array #full_plim_array * solar_system.max_power_charge
    #store it for later:
    df_pow_profile["Smart Charging"] = full_day_max_charging_power_profile_array #full_plim_array




    #and run the simulation of the system with the loaded datas:
    solar_system.run_storage_simulation(print_res=False)


    #and retrieve the results for grid power and inject it in the dataframe:
    df_pow_profile["Grid with storage"] =  solar_system.net_grid_balance_profile

    #The losses due to PV injection limitation is
    df_pow_profile["PV curtailment"] = solar_system.lostproduction_profile
    curtailment_lost_energy_kWh =df_pow_profile["PV curtailment"].sum() * timestep


    battery_power_array = solar_system.clamped_batt_pow_profile
    df_pow_profile["Battery power"] = battery_power_array 
    #separate the positive and the negative power to compute charge and discharge energy:
    df_pow_profile["Battery discharge power only"] = df_pow_profile["Battery power"].mask(df_pow_profile["Battery power"] > 0, 0.0)
    df_pow_profile["Battery charge power only"] = df_pow_profile["Battery power"] .mask(df_pow_profile["Battery power"] < 0, 0.0)

    # Replace all negative values with 0 to have the consumption only
    df_pow_profile["Grid consumption with storage"] = df_pow_profile["Grid with storage"].mask(df_pow_profile["Grid with storage"] < 0, 0.0)
    # Replace all positive values with 0 to have the injection only
    df_pow_profile["Grid injection with storage"] = df_pow_profile["Grid with storage"].mask(df_pow_profile["Grid with storage"] > 0, 0.0)

    soc_array = solar_system.soc_profile
    df_pow_profile["SOC"] = soc_array 

    #and compute the price with that power profile
    df_pow_profile["CostForBuyingWithStorage"] = (df_pow_profile["Grid consumption with storage"] * df_pow_profile["price buy"] * timestep)   #note, result is true/false, and .astype(int) convert to 1/0
    cost_buying_solar_storage_chf = df_pow_profile["CostForBuyingWithStorage"].sum()
    #print("Price paid with storage:", cost_buying_solar_storage_chf)
    grid_consumption_kWh_with_storage = df_pow_profile["Grid consumption with storage"].sum() * timestep

    df_pow_profile["SellSolarWithStorage"] = (-df_pow_profile["Grid injection with storage"] * df_pow_profile["price sell PV"] * timestep)   
    sellings_solar_storage_chf = df_pow_profile["SellSolarWithStorage"].sum()
    #print("Sold PV electricity with with solar only, no storage:", sellings_solar_storage_chf)
    grid_injection_kWh_with_storage = -df_pow_profile["Grid injection with storage"].sum() * timestep

    #bilan batterie  
    delta_e_batt=(soc_array[-1]-soc_array[0])/100.0*battery_size_kwh_usr_input  #last SOC - first SOC of the simulation
    #valorisé au prix moyen de la journée:
    mean_price_with_storage = cost_buying_solar_storage_chf/grid_consumption_kWh_with_storage
    storage_value = delta_e_batt * mean_price_with_storage # np.mean(cost_normal_profile_with_vario_with_storage/consumption_kWh)




    #*********************
    # Computations for the peak power

    peak_power_of_production = df_pow_profile["Solar power scaled"].max()

    peak_grid_consumption_with_solar = df_pow_profile["grid consumption reference"].max()
    peak_grid_injection_with_solar = df_pow_profile["grid injection reference"].min()

    peak_grid_consumption_with_batteries = df_pow_profile["Grid consumption with storage"].max()
    peak_grid_injection_with_batteries = df_pow_profile["Grid injection with storage"].min()




    #*********************
    # Computations of the bills
    peak_price_usr_input = 0.0  #TODO

    bill_of_peak_without_nothing = peak_power_of_consumption * peak_price_usr_input
    bill_of_peak_solar_only = peak_grid_consumption_with_solar * peak_price_usr_input
    bill_of_peak_with_storage = peak_grid_consumption_with_batteries*peak_price_usr_input

    bill_without_nothing = cost_buying_no_solar_chf + bill_of_peak_without_nothing
    bill_with_solar_only = cost_buying_solar_only_chf -  sellings_solar_only_chf + bill_of_peak_solar_only
    bill_with_storage = cost_buying_solar_storage_chf - sellings_solar_storage_chf + bill_of_peak_with_storage


    gain_of_storage = bill_with_solar_only - bill_with_storage
    total_gain_of_solar_and_storage= bill_without_nothing - bill_with_storage



    #*********************
    # Analyse the results

    #self_consumption_ratio_with_storage = (scaled_production_kWh-grid_injection_kWh_with_storage-curtailment_lost_energy_kWh) / scaled_production_kWh * 100
    #print(" Check selfconsumption: " , self_consumption_ratio_with_storage, solar_system.selfconsumption_rate)
    self_consumption_ratio_with_storage = solar_system.selfconsumption_rate
    #autarky_ratio_with_storage = (consumption_kWh-grid_consumption_kWh_with_storage) / consumption_kWh * 100.0  
    #print(" Check: " , autarky_ratio_with_storage, solar_system.autarky_rate)
    autarky_ratio_with_storage = solar_system.autarky_rate


    #save hours sampling for heatmap display:
    hours_mean_df = df_pow_profile.resample('h', label="right", closed="right").mean() 
    day_kwh_df = hours_mean_df.resample('d').sum() 
    month_kwh_df = day_kwh_df.resample('ME').sum() 

    batt_throughput_energy = -month_kwh_df['Battery discharge power only'].sum()
    if battery_size_kwh_usr_input == 0.0:
        equivalent_80percent_cycles = 0.0
        cost_of_stored_kWh_over_15_years = 0.0
    else :
        equivalent_80percent_cycles =  batt_throughput_energy / battery_size_kwh_usr_input / 0.8

        #assuming the data are always for 1 full year: TODO make it in function of the data size
        cost_of_stored_kWh_over_15_years = batt_total_cost_usr_input / 15.0 / batt_throughput_energy

    #st.write("📋 **Données de production et consommation**")
    
    st.write("📋 **Reférence sans solaire ni stockage, achat au réseau simple 🔌**")

    col1, col2 = st.columns(2)
    col1.metric("Consommation", str(int(consumption_kWh))+" kWh")
    col2.metric("Facture", str(int(bill_without_nothing))+" CHF")
    



    st.write("📋 **Résultat avec solaire seulement ☀️, sans stockage**")
    #col1, col2 = st.columns(2)
    st.metric("Production", str(int(scaled_production_kWh))+" kWh")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Achat au réseau", str(int(reference_grid_consumption_kWh)) + " kWh" , f"{(reference_grid_consumption_kWh-consumption_kWh)/consumption_kWh*100 :.1f}" + "%", delta_color="off")
    col2.metric("Revente suplus solaire", str(int(reference_grid_injection_kWh)) + " kWh")
    col3.metric("Autonomie", f"{reference_autarky_ratio :.1f}" + "%")
    col4.metric("Auto-consommation", f"{reference_self_consumption_ratio :.1f}" + "%")
    col5.metric("Facture", f"{bill_with_solar_only :.0f}" + "CHF", f" { bill_with_solar_only-bill_without_nothing :.0f}"+"CHF", delta_color="off" )

    st.write("📋 **Résultat avec solaire  ☀️ et stockage 🔋** ")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Achat au réseau", str(int(grid_consumption_kWh_with_storage))+" kWh", f"{(grid_consumption_kWh_with_storage-reference_grid_consumption_kWh)/reference_grid_consumption_kWh*100 :.1f}" + "%", delta_color="off")
    col2.metric("Revente suplus solaire", str(int(grid_injection_kWh_with_storage))+" kWh", f"{(grid_injection_kWh_with_storage-reference_grid_injection_kWh)/reference_grid_injection_kWh*100 :.1f}" + "%", delta_color="off")
    col3.metric("Autonomie", f"{(autarky_ratio_with_storage) :.1f}" + "%", f"{(autarky_ratio_with_storage-reference_autarky_ratio) :.1f}"+"%")
    col4.metric("Auto-consommation", f"{self_consumption_ratio_with_storage :.1f}"+"%", f"{(self_consumption_ratio_with_storage-reference_self_consumption_ratio) :.1f}" + "%" )
    #col3.metric("Self-consumption", f"{(self_consumption_ratio_with_storage) :.1f, }"+"%", f"{(self_consumption_ratio_with_storage-reference_self_consumption_ratio) :.1f}"+"%")
    col5.metric("Facture", f" {bill_with_storage :.0f}"+"CHF", f" { bill_with_storage-bill_with_solar_only :.0f}"+"CHF", delta_color="off" )



    #st.markdown("---")



                


    # fig_polar_indicators2 = build_monthly_indicators_polar_figure(month_kwh_df)
    # st.pyplot(fig_polar_indicators2)
    st.write( "\n \n")


    fig_polar_indicators = build_daily_indicators_polar_fraction_figure(day_kwh_df)
    st.pyplot(fig_polar_indicators)

    st.write(""" Des explication sur les indicateurs utilisés peuvent être trouvés ici --> [INDICATEURS DANS LE SOLAIRE](https://autoconsommation.ch/indicateurs/)  """)
    st.write( "\n \n")



    st.write(" **Les résultats**")



    st.markdown(f""" ***Référence***
    - La consommation d'électricité pour cette période est {consumption_kWh:.2f} kWh 🔌
    - Le coût de l'électricité du réseau sans panneaux solaires est {cost_buying_no_solar_chf:.2f} CHF, prix moyen est {cost_buying_no_solar_chf/consumption_kWh:.3f} CHF/kWh
    """)

    st.markdown(f""" ***☀️ Avec solaire***
    - La consommation d'électricité sur le réseau pour cette période est {reference_grid_consumption_kWh:.2f} kWh avec des panneaux solaires
    - Le coût de l'électricité du réseau est {cost_buying_solar_only_chf:.2f} CHF avec des panneaux solaires, prix moyen est {cost_buying_solar_only_chf/reference_grid_consumption_kWh:.3f} CHF/kWh
    - L'énergie perdue due à la limitation de revente sur le réseau est {reference_curtailment_lost_energy_kwh :.0f} kWh et le niveau de limitation est {pv_injection_curtailment_power:.2f} kW
    - La revente de l'électricité PV est {sellings_solar_only_chf:.2f} CHF avec des panneaux solaires, prix moyen est {sellings_solar_only_chf/reference_grid_injection_kWh:.3f} CHF/kWh
    - La facture totale est {bill_with_solar_only:.2f} CHF avec des panneaux solaires, un gain de {cost_buying_no_solar_chf-bill_with_solar_only:.1f} CHF grâce aux panneaux solaires
    - Le prix investi dans les panneaux solaires de {PV_total_cost_usr_input:.2f} CHF est retrouvé en {PV_total_cost_usr_input/(cost_buying_no_solar_chf-bill_with_solar_only):.1f} années (calcul simple sans actualisation, si une année complète de données est utilisée).
    """)

    st.markdown(f""" ***🔋 Avec stockage***
    - La consommation d'électricité sur le réseau pour cette période est {grid_consumption_kWh_with_storage:.2f} kWh avec stockage
    - L'énergie perdue due à la limitation de revente sur le réseau est {curtailment_lost_energy_kWh :.0f} kWh
    - Le coût de l'électricité du réseau est {cost_buying_solar_storage_chf:.2f} CHF avec stockage, prix moyen est {cost_buying_solar_storage_chf/grid_consumption_kWh_with_storage:.3f} CHF/kWh
    - La revente de l'électricité PV est {sellings_solar_storage_chf:.2f} CHF avec stockage, prix moyen est {sellings_solar_storage_chf/grid_injection_kWh_with_storage:.3f} CHF/kWh
    - La facture totale est {bill_with_storage:.2f} CHF avec stockage, un gain de {bill_with_solar_only - bill_with_storage :.1f} CHF grâce au stockage
    - Le prix investi dans le solaire et les batteries de {(PV_total_cost_usr_input + batt_total_cost_usr_input) :.2f} CHF est retrouvé en {(PV_total_cost_usr_input + batt_total_cost_usr_input) / (cost_buying_no_solar_chf - bill_with_storage) :.1f} années (calcul simple sans actualisation, si une année complète de données est utilisée).
    - **Gain TOTAL** avec solaire + stockage est {cost_buying_no_solar_chf - bill_with_storage :.2f} CHF """)


    st.write(f" Note: La facture est calculée avec les kWh seulement, sans les abonnements. Ce total peut différer de la facture réelle mais les abonnements sont les mêmes dans la plupart des cas. ")

    st.write( "\n")
    st.write( "\n")


            
    fig_merged = px.line(
        df_pow_profile,
        x=df_pow_profile.index,  # ← index temps
        y=[ "Solar power scaled", "Consumption [kW]"],
        labels={
            "index": "Temps",
            "value": "Puissance [kW]",
            "variable": "Série",
        },
        color_discrete_sequence=["orange", "lightblue"],
        title="🌞 Production PV estimée -⚡Consommation (smart-meter)",
    )
    #renommer les legendes en français
    fig_merged.data[0].name = "Production solaire"
    fig_merged.data[1].name = "Consommation"

    st.plotly_chart(fig_merged, width='stretch')



    # Energy Consumption Plot using Plotly
    fig_simstorage_profile = px.line(df_pow_profile, 
                            x=df_pow_profile.index, 
                            y=[ "Consumption [kW]","Grid consumption with storage","Grid injection with storage"], 
                            title="⚡🔌 Consommation depuis le réseau avec solaire et stockage", 
                            labels={"value": "Puissance (kW)", "variable": "Legend"},
    )
    
    #renommer les legendes en français
    fig_simstorage_profile.data[0].name = "Consommation"
    fig_simstorage_profile.data[1].name = "Achat au réseau"
    fig_simstorage_profile.data[2].name = "Injection réseau"
   
    # Move legend below the graph
    fig_simstorage_profile.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Position below the graph
            xanchor="center",
            x=0.1
        )
    )
    st.plotly_chart(fig_simstorage_profile)


    with st.expander("Aperçu des heatmaps prod et conso"):

        st.write("Les données vues sous forme de heatmap, ici chaque heure de consommation/production de l'année est affichée, organisée par jour de l'année et heure du jour.")

        fig_production_heatmap = build_production_heatmap_figure(hours_mean_df)
        st.pyplot(fig_production_heatmap)

        fig_consumption_heatmap = build_consumption_heatmap_figure(hours_mean_df)
        st.pyplot(fig_consumption_heatmap)
    


    with st.expander("Aperçu du fonctionnement de la batterie"):
        
        # Battery Plot using Plotly
        fig_soc_profile = px.area(df_pow_profile, 
                                x=df_pow_profile.index, 
                                y=[ "SOC"], 
                                title=" 🔋 Etat de charge de la batterie", 
                                labels={"index": "Temps", "value": "Etat de charge (%)", "variable": "SOC"},
                                color_discrete_sequence = ["green"]
        )
            
        # Move legend below the graph
        fig_soc_profile.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,  # Position below the graph
                xanchor="center",
                x=0.1
            )
        )
        st.plotly_chart(fig_soc_profile)




        st.write(f"L'énergie transitant (throughout) dans la batterie est de {batt_throughput_energy :.0f} kWh et cela est équivalent à {equivalent_80percent_cycles :.0f} cycles à 80% DOD")

        st.write(f"Si il y a une année de données, sur 15 ans, cela donne un coût de stockage de {cost_of_stored_kWh_over_15_years*100.0 :.1f} ct/kWh, plus la valeur de l'énergie solaire (avec le prix de vente au réseau ici), le coût de l'énergie stockée est {(cost_of_stored_kWh_over_15_years*100.0 + fixed_price_sell_usr_input*100.0 ) :.0f}  ct/kWh")


        fig_bat_inout = build_bat_inout_figure(day_kwh_df, month_kwh_df)
        st.pyplot(fig_bat_inout)



        # Energy Consumption Plot using Plotly
        fig_batt_profile = px.area(df_pow_profile, 
                                x=df_pow_profile.index, 
                                y=[ "Battery power"], 
                                title=" 🔋 Puissance de la batterie", 
                                labels={"index": "Temps", "value": "Battery power [kW]", "variable": "Legend"}
        )


        # Move legend below the graph
        fig_batt_profile.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,  # Position below the graph
                xanchor="center",
                x=0.1
            )
        )
        st.plotly_chart(fig_batt_profile)
        

        st.markdown(f""" ***🔎 Quelques détails***
                    
    - L'état de charge de la batterie au début ({soc_array[0]:.0f} %) et à la fin de la période ({soc_array[-1]:.0f} %) sont différents, soit {delta_e_batt:.1f} kWh, à compter dans le prix final.
    - La valeur de l'énergie stockée restante dans la batterie avec le prix moyen est {storage_value:.2f} CHF
    - La consommation propre du système de stockage est de {INVERTER_STANDBY_W: .1f} watts si il y en a une, c'est {INVERTER_STANDBY_W*24/1000} kWh/day. Il y a un rendement de 0.95 compté.
    - Dans cette simulation le contrôle de la batterie se fait comme la plupart des onduleurs : charger dès qu'il y a un excédent solaire et décharger dès qu'il n'y a pas assez de solaire. 
      Cela n'est pas très intelligent et ne permet pas d'optimiser des cas spéciaux comme les tarifs dynamiques, ... pour l'instant    """)
        
        

    #fig_polar_consumption = build_polar_consumption_profile(df_pow_profile)
    #st.pyplot(fig_polar_consumption)



    # with st.expander("Aperçu des données fusionnées (premières lignes)"):
    #     st.dataframe(df_pow_profile.head(200), width='stretch')

