
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configuración de logs para trazabilidad en consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


# ==============================================================================
# PASO 1: CONFIGURACIÓN ESPECÍFICA DE ESTE INFORME
# ==============================================================================

CONFIG = {
    # 1.1. Rutas de archivos
    "ruta_origen": "origen_datos.csv",          # Ruta al archivo descargado (.csv, .xlsx, .txt)
    "ruta_destino": "Plantilla_Qlik.xlsx",      # Ruta al Excel que lee Qlik
    "hoja_destino": "Datos",                    # Pestaña exacta del Excel destino
    
    # 1.2. Propiedades de lectura del archivo origen
    "lectura": {
        "tipo": "csv",               # 'csv', 'txt' o 'excel'
        "separador": ";",            # ';', ',', '\t' o None (autodetección)
        "encoding": "utf-8",         # 'utf-8', 'latin-1' o 'cp1252' (para tildes/ñ)
        "decimal": ",",              # ',' para formato español ("15,50"), '.' para inglés ("15.50")
        "miles": ".",                # '.' para formato español ("1.000"), ',' para inglés
        "filas_a_saltar": 0,         # Filas de títulos vacíos antes de la cabecera real
        "hoja_origen": 0             # Solo si el origen es Excel (índice 0 o nombre de pestaña)
    },
    
    # 1.3. Mapeo de nombres: { "Nombre_Columna_En_Origen": "Nombre_Esperado_Por_Qlik" }
    "mapeo_columnas": {
        "COD_LOCAL": "id_tienda",
        "FECHA_VENTA": "fecha",
        "IMPORTE_NETO": "importe_neto",
        "CANTIDAD": "unidades",
        "CANAL_VENTA": "canal"
    },
    
    # 1.4. Lista blanca de columnas finales (en el orden exacto que espera Qlik)
    # Toda columna del origen que NO esté en esta lista será eliminada automáticamente.
    "columnas_ordenadas": [
        "id_tienda",
        "fecha",
        "importe_neto",
        "unidades",
        "canal"
    ],
    
    # 1.5. Definición de tipos de datos clave
    "columnas_como_texto": ["id_tienda"],        # Evita que '00123' se convierta en 123
    "columnas_numericas": ["importe_neto", "unidades"],
    "columnas_fecha": {
        "fecha": "%d/%m/%Y"                      # Ajustar formato: '%Y-%m-%d', '%d/%m/%Y', etc.
    }
}


# ==============================================================================
# PASO 2: INGESTA / LECTURA ROBUSTA
# ==============================================================================
def leer_datos(config: dict) -> pd.DataFrame:
    """Lee el archivo de origen gestionando encodings, separadores y saltos."""
    ruta = Path(config["ruta_origen"])
    cfg = config["lectura"]
    
    if not ruta.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo origen en: {ruta.resolve()}")
    
    logging.info(f"📂 [Paso 2] Leyendo archivo: {ruta.name}")
    tipo = cfg.get("tipo", ruta.suffix.replace(".", "")).lower()
    
    if tipo in ["csv", "txt"]:
        df = pd.read_csv(
            ruta,
            sep=cfg.get("separador", ";"),
            engine="python" if cfg.get("separador") is None else "c",
            encoding=cfg.get("encoding", "utf-8"),
            skiprows=cfg.get("filas_a_saltar", 0),
            dtype=str  # Leemos inicialmente todo como string para evitar corrupciones automáticas
        )
    elif tipo in ["excel", "xlsx", "xls"]:
        df = pd.read_excel(
            ruta,
            sheet_name=cfg.get("hoja_origen", 0),
            skiprows=cfg.get("filas_a_saltar", 0),
            dtype=str
        )
    else:
        raise ValueError(f"Formato de archivo no soportado: {tipo}")
        
    logging.info(f"   Filas leídas: {len(df):,} | Columnas detectadas: {len(df.columns)}")
    return df


# ==============================================================================
# PASO 3: LIMPIEZA DE ESTRUCTURA (Cabeceras y Filtro de Columnas)
# ==============================================================================
def estructurar_columnas(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Limpia nombres de columnas, renombra y descarta las innecesarias."""
    logging.info("🧹 [Paso 3] Limpiando y renombrando columnas...")
    df = df.copy()
    
    # 3.1. Eliminar espacios accidentales en los nombres de las columnas (' FECHA ' -> 'FECHA')
    df.columns = df.columns.astype(str).str.strip()
    
    # 3.2. Renombrar según el diccionario de mapeo
    df = df.rename(columns=config["mapeo_columnas"])
    
    # 3.3. Validar que las columnas requeridas existen
    cols_esperadas = config["columnas_ordenadas"]
    cols_faltantes = [c for c in cols_esperadas if c not in df.columns]
    if cols_faltantes:
        raise KeyError(f"❌ Error: Faltan las siguientes columnas en el origen: {cols_faltantes}")
        
    # 3.4. Filtrar y ordenar: descartamos todas las columnas que no nos interesen
    df = df[cols_esperadas]
    logging.info(f"   Estructura final: {len(df.columns)} columnas seleccionadas.")
    return df


# ==============================================================================
# PASO 4: TRANSFORMACIÓN Y NORMALIZACIÓN DE CONTENIDO
# ==============================================================================
def transformar_contenido(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Limpia formatos numéricos (comas/puntos), fechas y textos."""
    logging.info("⚙️  [Paso 4] Aplicando transformaciones de tipos y formatos...")
    df = df.copy()
    cfg_lec = config["lectura"]
    dec = cfg_lec.get("decimal", ",")
    mil = cfg_lec.get("miles", ".")
    
    # 4.1. Limpieza de textos (quitar espacios en blanco al inicio/final de cada celda)
    for col in df.columns:
        if col not in config["columnas_numericas"]:
            df[col] = df[col].astype(str).str.strip()
            # Tratar representaciones textuales de nulos
            df[col] = df[col].replace({"nan": None, "None": None, "": None, "NULL": None})

    # 4.2. Corrección y conversión de columnas numéricas (manejo de '1.250,50' -> 1250.50)
    for col in config.get("columnas_numericas", []):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(mil, "", regex=False)    # Quita separador de miles
                .str.replace(dec, ".", regex=False)   # Reemplaza coma decimal por punto estándar
                .str.replace("€", "", regex=False)    # Quita posibles símbolos de moneda
                .str.replace("%", "", regex=False)    # Quita posibles símbolos de porcentaje
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce") # Convierte a float/int (nulos si falla)

    # 4.3. Formateo de fechas
    for col, formato in config.get("columnas_fecha", {}).items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=formato, errors="coerce")

    # 4.4. Columnas identificadoras como texto puro (para no perder ceros: '0045')
    for col in config.get("columnas_como_texto", []):
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None})

    return df


# ==============================================================================
# PASO 5: VALIDACIÓN DE CALIDAD DE DATOS
# ==============================================================================
def validar_datos(df: pd.DataFrame) -> None:
    """Chequeos rápidos de sanidad antes de escribir el archivo final."""
    logging.info("🔍 [Paso 5] Validando calidad de los datos...")
    
    # 5.1. Validar que no esté vacío
    if df.empty:
        raise ValueError("❌ El DataFrame resultante está completamente vacío.")
        
    # 5.2. Alerta de registros nulos en campos críticos (puedes personalizar las columnas)
    for col in df.columns:
        nulos = df[col].isna().sum()
        if nulos > 0:
            pct = (nulos / len(df)) * 100
            logging.warning(f"   ⚠️  Columna '{col}' tiene {nulos:,} valores nulos ({pct:.1f}%).")
            
    logging.info("   Validación completada.")


# ==============================================================================
# PASO 6: CARGA / EXPORTACIÓN AL EXCEL DE QLIK
# ==============================================================================
def exportar_resultado(df: pd.DataFrame, config: dict) -> None:
    """Sobreescribe la pestaña del Excel de destino manteniendo el resto del libro."""
    ruta = Path(config["ruta_destino"])
    hoja = config["hoja_destino"]
    
    logging.info(f"💾 [Paso 6] Exportando {len(df):,} filas a '{ruta.name}' -> Hoja: '{hoja}'")
    
    if ruta.exists():
        # Si el Excel ya existe, reemplazamos solo la pestaña especificada
        with pd.ExcelWriter(ruta, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=hoja, index=False)
    else:
        # Si no existe, creamos el archivo desde cero
        with pd.ExcelWriter(ruta, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=hoja, index=False)
            
    logging.info("🚀 ¡Proceso completado con éxito!")


# ==============================================================================
# EJECUCIÓN DEL PIPELINE
# ==============================================================================
def ejecutar_pipeline():
    try:
        logging.info("=== INICIO DEL PROCESO DE ACTUALIZACIÓN ===")
        
        # Flujo secuencial
        df_raw = leer_datos(CONFIG)
        df_cols = estructurar_columnas(df_raw, CONFIG)
        df_clean = transformar_contenido(df_cols, CONFIG)
        validar_datos(df_clean)
        exportar_resultado(df_clean, CONFIG)
        
        logging.info("=== PROCESO FINALIZADO SATISFACTORIAMENTE ===")
        
    except Exception as error:
        logging.error(f"❌ ERROR CRÍTICO EN EL PIPELINE: {str(error)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    ejecutar_pipeline()