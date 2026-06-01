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
- `ROI_POLYGON`: ROI in JSON, ad esempio `[[0.50,0.50],[0.71,0.45],[0.77,0.63],[0.57,0.69]]`.
- `ROTATION_ANGLE`: angolo di rotazione manuale in gradi per raddrizzare il frame. Per il video di esempio il default è `-45`.
- `AUTO_ROTATE`: `true|false` — se `true` il tracker proverà a stimare automaticamente l'angolazione dalla prima immagine. Default `false`.
- `FRAME_STRIDE`: processa un frame ogni N letture del video. Default `2`.
- `CONFIDENCE`: soglia di confidenza YOLO. Default `0.20`.
- `IMGSZ`: dimensione d'ingresso YOLO. Default `416`.

Esempio:

```bash
set VIDEO_PATH=res/Export 01-06-2026 10-18-50.MP4
set YOLO_MODEL=yolov8n.pt
set RALLA_CLASS_ID=0
set ROTATION_ANGLE=-45
set FRAME_STRIDE=2
set CONFIDENCE=0.20
set IMGSZ=416
python app.py
```

Esempio con rotazione manuale (PowerShell):

```powershell
$env:ROTATION_ANGLE = "-45"
$env:AUTO_ROTATE = "false"
python app.py
```

## Note tecniche

- La ROI viene disegnata e usata come filtro di conteggio: i mezzi vengono considerati solo se il centroide cade dentro il poligono.
- `Sbarco` viene contato se il mezzo attraversa la ROI da destra verso sinistra; in caso contrario viene contato come `Imbarco`.
- Il tracker mostra overlay con ROI, bounding box, ID, FPS, numero detection nella ROI e numero di track attivi.
- Il carico è ridotto con `FRAME_STRIDE` e con un input YOLO più piccolo (`IMGSZ=416`).
