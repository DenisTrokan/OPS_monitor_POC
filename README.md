# OPS Monitor POC

Proof of Concept per il monitoraggio di ralle in un terminal Ro-Ro usando YOLOv8 + tracking.

## Avvio rapido

1. Crea e attiva un virtual environment.
2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

3. Avvia il server:

```bash
python app.py
```

4. Apri il browser su `http://localhost:5000`.

> Nota: il primo avvio scarica automaticamente i pesi YOLOv8.

## Configurazione

Variabili ambiente supportate:

- `VIDEO_PATH`: percorso del file video (default: `res/Export 01-06-2026 10-18-50.MP4`)
- `YOLO_MODEL`: pesi YOLOv8 (default: `yolov8n.pt`)
- `RALLA_CLASS_ID`: ID della classe da trattare come "Ralla" (default: `0`)
- `LINE_RATIO`: posizione della linea virtuale (0.0-1.0, default: `0.5`)

Esempio:

```bash
set VIDEO_PATH=res/Export 01-06-2026 10-18-50.MP4
set YOLO_MODEL=yolov8n.pt
set RALLA_CLASS_ID=0
set LINE_RATIO=0.5
python app.py
```

## Note tecniche

- La linea virtuale viene impostata in base all'altezza del frame (vedi commenti in `tracker.py`).
- Il tracking usa ByteTrack integrato in YOLOv8 per assegnare un ID stabile a ogni ralla.
- Lo stream video e i KPI vengono aggiornati in tempo reale tramite Flask.
