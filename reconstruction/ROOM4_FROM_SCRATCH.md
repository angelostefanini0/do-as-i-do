# Room4 da zero: Aria → EgoAllo + sedia 3D → ECHO

Questa guida documenta il **run whitechair Room4 del 29 settembre 2026**, non una
pipeline universale per qualsiasi VRS. È pensata per una nuova macchina Linux
con GPU NVIDIA: indica i sorgenti, gli asset non versionati, i comandi, i punti
di controllo e gli errori che abbiamo incontrato. I modelli/dataset soggetti a
licenza o accesso controllato **non sono redistribuiti nel fork**.

> **Stato dei sorgenti, prima di iniziare.** I fork pubblicati contengono
> `do-as-i-do` branch `aria-room4-object-tracking` (il codice e il pin del
> submodule sono nel commit `e441c3f`) e il suo
> submodule `Fast-SAM3D` (commit `0f10267`). Alla data di questa guida, le
> modifiche Room4 a **EgoAllo e ECHO esistono solo nei worktree locali**:
> clonare soltanto `brentyi/egoallo` e `ptrvilya/echo` non basta. Occorre
> trasferire il codice locale o pubblicare prima due ulteriori fork. La
> sezione «Sorgenti non ancora pubblicati» elenca i file. Non dichiarare
> riprodotto il run 017 se quel passaggio manca.

## 1. Cosa serve davvero

| Asset | Origine e accesso | Dove deve finire | Necessario per |
| --- | --- | --- | --- |
| VRS **`Video_moving_around_room_4_20260928_232041.vrs`** | Registrazione Aria originale, **non** un dataset pubblico sostitutivo: trasferirla dalla macchina sorgente, se autorizzati | `$ARIA/<nome>.vrs` | entrambi i rami |
| MPS della **stessa** registrazione: `slam/closed_loop_trajectory.csv`, `slam/semidense_points.csv.gz`, `hand_tracking/hand_tracking_results.csv` | Trasferire l'output originale, oppure richiedere MPS con il proprio account seguendo la [guida Project Aria MPS](https://facebookresearch.github.io/projectaria_tools/gen2/ark/mps/start); non mischiare VRS/MPS di registrazioni diverse | `$ARIA/mps_<stem>_vrs/{slam,hand_tracking}/` | pose camera, scala sedia, polsi EgoAllo |
| EgoAllo checkpoint `egoallo_checkpoint_april13/checkpoints_3000000/model.safetensors` | [EgoAllo README](https://github.com/brentyi/egoallo#readme), [checkpoint](https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view) | `$EGO/egoallo_checkpoint_april13/` | EgoAllo; gli 8 GB di example trajectories sono opzionali |
| SMPL-H neutral Extended, 16 shape parameters | [MANO/SMPL+H](https://mano.is.tue.mpg.de/), download/licenza manuali | `$EGO/data/smplh/neutral/model.npz` | EgoAllo e preprocess ECHO |
| HaMeR checkpoint, ViTPose, MANO_RIGHT | [HaMeR ufficiale](https://github.com/geopavlakos/hamer), [hamer_helper](https://github.com/brentyi/hamer_helper) | `$EGO/third_party/hamer/_DATA/...` secondo README HaMeR | `hamer_outputs.pkl` usato nel run `aria_hamer` |
| SAM 3 `sam3.pt` | [facebook/sam3](https://huggingface.co/facebook/sam3), accesso gated | es. `$RECON/weights/sam3/sam3.pt` | maschere della sedia |
| SAM 3D Objects checkpoint e YAML | [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), accesso gated; `reconstruction/setup/02_fetch_weights.sh` | `$RECON/weights/sam3d_shared/hf/` e link in `modules/{sam-3d-objects,Fast-SAM3D}/checkpoints/hf/` | mesh singola di riferimento |
| BootsTAPIR v2 | [checkpoint pubblico](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt) | `$RECON/weights/tapnet/bootstapir_checkpoint_v2.pt` | landmark della sedia |
| MoGe v1 e DINOv2 | `Ruicheng/moge-vitl` e `facebookresearch/dinov2`, scaricati alla prima inizializzazione della mesh | cache Hugging Face / Torch Hub dell'utente che esegue `sam3d` | SAM3D; preparare la cache **prima** del runner offline |
| ECHO pretrained `echo.pth` | [link del README ECHO](https://nc.mlcloud.uni-tuebingen.de/index.php/s/dStmr3y6gTNte3D) | `$ECHO/assets/echo.pth` | run 016/017/019 |
| SMPL-X `SMPLX_{NEUTRAL,MALE,FEMALE}.npz` | [download ufficiale SMPL-X](https://github.com/vchoutas/smplx#downloading-the-model), licenza manuale | `$ECHO/body_models/smplx/` | ECHO e video |
| Mesh OMOMO `whitechair_cleaned_simplified.obj` | [OMOMO release](https://github.com/lijiaman/omomo_release) | `$ECHO/data/raw/omomo/data/captured_objects/` | mesh canonica ECHO e condizionamento 017 |

Il clone ECHO include già `assets/objects/omomo/{canonical_object_transform.pkl,object_keypoints/whitechair.npz,object_pointnext.pkl}` e gli asset corrispondenti BEHAVE: per **questo** run non occorre scaricare l'intero BEHAVE. Il preprocess legge però la mesh OMOMO grezza. AMASS, le annotazioni complete OMOMO e BEHAVE sono necessari per riaddestramento/benchmark ECHO, **non** per generare 017/020 dal checkpoint: vedere [ECHO `docs/data.md`](https://github.com/ptrvilya/echo/blob/main/docs/data.md) per gli eventuali download ufficiali e le licenze. Non inserire VRS, MPS, modelli SMPL, checkpoint o dataset nel fork Git.

Se si vuole **anche** ricostruire i dataset di addestramento/evaluation ECHO:
da [AMASS](https://amass.is.tue.mpg.de/) servono le subset SMPL-X neutral
elencate in `ECHO/docs/data.md`; da [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/license.html)
servono scansioni oggetti, annotazioni 30 fps e `split.json`; da OMOMO servono
anche `train/test_diffusion_manip_seq_joints24.p`. Non confondere questi
download opzionali con gli input minimi del run Room4, e non tentare il
preprocessing completo BEHAVE senza anche SMPLFitter e i suoi body model.

## 2. Layout portabile e versioni del codice

Negli esempi sostituire `/path/to/work` con una directory reale, senza spazi se
possibile. Il nome **esatto** del VRS è importante: il runner riconosce questa
sequenza tramite lo stem e solo allora sceglie il tracker metrico corretto.

```bash
export WORK=/path/to/work
export RECON="$WORK/do-as-i-do/reconstruction"
export EGO="$WORK/egoallo"
export ECHO="$WORK/echo"
export ARIA="$WORK/AriaData"
export STEM=Video_moving_around_room_4_20260928_232041
export VRS="$ARIA/$STEM.vrs"
export MPS="$ARIA/mps_${STEM}_vrs"
mkdir -p "$WORK"

GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  --branch aria-room4-object-tracking \
  https://github.com/angelostefanini0/do-as-i-do.git "$WORK/do-as-i-do"
git -C "$WORK/do-as-i-do" log -1 --oneline     # branch della guida
git -C "$WORK/do-as-i-do" submodule status     # Fast-SAM3D: 0f10267...

git clone https://github.com/brentyi/egoallo.git "$EGO"
git -C "$EGO" switch --detach 0c20ae8c8d48be4ff682803ea523bcb00a6a708d
git clone https://github.com/ptrvilya/echo.git "$ECHO"
git -C "$ECHO" switch --detach dd46d386eb800a2cbf687de1ee21c7614a2046eb
```

### Sorgenti non ancora pubblicati: **passaggio obbligatorio**

Prima di installare EgoAllo/ECHO, copiare dalla vecchia macchina **sopra i due
clone, senza cancellare file** oppure pubblicare due fork che contengano:

- EgoAllo: `pyproject.toml` (Project Aria Tools 2.3.0 invece dei pin 1.6.0),
  `src/egoallo/hand_detection_structs.py` (parser MPS Gen2
  `hand_tracking_results.csv` e retrocompatibilità Gen1), `.gitignore`.
- ECHO, file modificati: `.gitignore`, `echo/configs/{config,datasets}.py`,
  `echo/core/sampler.py`, `echo/data/{__init__,batch_data,dataset}.py`,
  `echo/model/{base,model_hoic}.py`, `scenarios/{echo,env}.yaml`.
- ECHO, file nuovi essenziali: `scenarios/aria_egoallo_{whitechair,tracked_whitechair}.yaml`,
  `scripts/{preprocess_egoallo_conditioning,build_aria_tracked_object_conditioning,render_sample,prepare_sam_contact_guidance,optimize_sam_object_contacts,evaluate_room4_object_silhouette,evaluate_sam_contact_guidance}.py`.

Per copiare il codice senza ambienti/dataset da una macchina raggiungibile via
SSH, un esempio (sostituire host e path sorgente) è:

```bash
export SOURCE_HOST=user@old-machine
export SOURCE_WORK=/path/to/old/workspace
rsync -a --exclude='/.git/' --exclude='/.mamba-env/' --exclude='/data/' \
  --exclude='/aria_trajectories/' --exclude='/egoallo_checkpoint_*/' \
  --exclude='/third_party/' \
  "$SOURCE_HOST:$SOURCE_WORK/egoallo/" "$EGO/"
rsync -a --exclude='/.git/' --exclude='/.mamba-env/' --exclude='/data/' \
  --exclude='/experiments/' --exclude='/body_models/' \
  --exclude='/assets/echo.pth' --exclude='/logs/' \
  "$SOURCE_HOST:$SOURCE_WORK/echo/" "$ECHO/"
```

**Non usare `--delete`.** Gli exclude con `/` iniziale valgono solo alla
radice, quindi non eliminano il package sorgente `src/egoallo/data/`. Poi
verificare che i file sopra siano presenti e
che `git diff` mostri le modifiche previste. I modelli/dati si trasferiscono
separatamente, nei path della tabella. Questo snapshot dei due worktree è un
limite di riproducibilità: fino alla pubblicazione dei loro commit non esiste
un comando `git clone` che lo sostituisca.

## 3. Ambienti e GPU

La macchina usata aveva RTX 4000 Ada (~20 GiB), driver compatibile CUDA 12.8,
Linux, `micromamba` e `ffmpeg`. La mesh SAM3D ha raggiunto ~19,7 GB di memoria
PyTorch allocata nel test: su GPU più piccole aspettarsi OOM. Gli ambienti
**non vanno fusi**:

| Ambiente | Stack osservato | Scopo |
| --- | --- | --- |
| `sam3` | Python 3.12, Torch 2.8 + cu128 | SAM3 video |
| `sam3d` | Python 3.11, Torch 2.8 + cu128 | SAM3D, calibrazione e PnP |
| `tapnet` | Python 3.10, Torch 2.7 + cu128 | BootsTAPIR |
| `$EGO/.mamba-env` | Python 3.12.14, JAX/JAXLIB 0.6.1, `projectaria-tools` 2.3.0 | VRS/HaMeR/EgoAllo e prima conversione HDF5 |
| ECHO (`echo`) | Python 3.10.21, Torch 2.9.1 | diffusione, contatti e rendering |

Nel repository `do-as-i-do`, [env/README.md](env/README.md) descrive i
conflitti già incontrati. I manifest `env/{sam3,sam3d,tapnet}.yml` sono export
di un ambiente funzionante, **non** garanzia di risoluzione su ogni macchina.
Su hardware CUDA 12.8 iniziare da quelli oppure creare `sam3` e `tapnet` con
`setup/01_create_envs.sh sam3` e `... tapnet`; il ramo `sam3d` dello script usa
pin upstream più vecchi e può richiedere adattamento. `hawor`/HaWoR **non è
necessario** per Room4: non confonderlo con **HaMeR** per le mani EgoAllo.

Mettere `micromamba` nel `PATH`. Impostare sul nuovo host `MICROMAMBA_BIN` al binario vero (`run_camera_aware_room4.sh`
ha come fallback un path della vecchia macchina) e `TAPIR_GPU=0` se c'è una
sola GPU. `config/paths.sh` imposta `CUDA_VISIBLE_DEVICES=0`. Verificare con
`micromamba run -n sam3d python -c 'import torch; print(torch.cuda.is_available())'`.

Per EgoAllo, **dopo** aver applicato la modifica a `pyproject.toml`:

```bash
micromamba create -y -p "$EGO/.mamba-env" python=3.12 pip
"$EGO/.mamba-env/bin/python" -m pip install -e "$EGO"
"$EGO/.mamba-env/bin/python" -m pip install 'jax[cuda12]==0.6.1' \
  'git+https://github.com/brentyi/jaxls.git'
```

Seguire inoltre il README di [HaMeR](https://github.com/geopavlakos/hamer)
e di [hamer_helper](https://github.com/brentyi/hamer_helper); qui HaMeR era
clonato in `$EGO/third_party/hamer` al commit `3a01849f4148352e9260b69bf28b65d1671a4905`
e installato editabile **nello stesso ambiente EgoAllo**. Prima del run
`2_run_hamer_on_vrs.py`, verificare almeno:
`third_party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`, relativo
`model_config.yaml`, `_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth` e
`_DATA/data/mano/MANO_RIGHT.pkl`. I download HaMeR/MANO seguono licenze e
istruzioni originali; la presenza del solo package Python non basta.

Per ECHO, seguire `environment.yml` del **codice locale modificato**:

```bash
cd "$ECHO"
conda env create -f environment.yml
conda activate echo
python -m pip install -e .
python -m pip install pyrender==0.1.45
```

Verificare `import cv2, h5py, pyrender, smplx, torch`; il renderer usa EGL
headless (`PYOPENGL_PLATFORM=egl`) e può richiedere librerie OpenGL/EGL di
sistema. In `scenarios/env.yaml` sostituire tutti i path assoluti della vecchia
macchina, soprattutto `env.smpl_all_models_folder` con `$ECHO/body_models` e
`env.smplfitter_data_folder` se si usa BEHAVE. Non usare un env ECHO per
`preprocess_egoallo_conditioning.py`: quel file importa il package EgoAllo.

## 4. Pesi oggetto e input Aria

1. Ottenere accesso Hugging Face a `facebook/sam3` e
   `facebook/sam-3d-objects`, poi `hf auth login`. Installare i pesi con
   `reconstruction/setup/02_fetch_weights.sh --download` oppure scaricare
   manualmente gli stessi file. Lo script scarica anche pesi HaWoR non usati
   qui e **non** scarica `sam3.pt`. Per quest'ultimo, ad esempio:

   ```bash
   hf download facebook/sam3 sam3.pt --local-dir "$RECON/weights/sam3"
   export SAM3_CHECKPOINT="$RECON/weights/sam3/sam3.pt"
   export SAM3D_CONFIG="$RECON/modules/Fast-SAM3D/checkpoints/hf/pipeline.yaml"
   ```

   Il `pipeline.yaml` deve trovare i relativi `.yaml/.ckpt` nella **stessa
   directory**. I symlink creati da `02_fetch_weights.sh` non devono puntare
   alla vecchia macchina. Lo script può riscrivere anche piccoli YAML dentro
   il submodule: dopo il download controllare
   `git -C "$RECON/modules/Fast-SAM3D" status --short` e non perdere la modifica
   versionata a `ss_encoder.yaml`. Posizionare BootsTAPIR nel path della tabella.
2. Prima del runner, popolare online la cache MoGe/DINOv2 richiesta da
   SAM3D. La configurazione usa `Ruicheng/moge-vitl` e
   `dinov2_vitl14_reg`; sulla vecchia macchina Torch Hub ha scaricato anche
   `dinov2_vitl14_reg4_pretrain.pth`. Un primo caricamento del modello SAM3D
   con rete abilitata, o il trasferimento autorizzato della cache, è il
   controllo più affidabile. Il runner imposta `HF_HUB_OFFLINE=1`, perciò
   **non** può recuperare un peso mancante durante la pipeline.
3. Trasferire VRS e MPS nello schema seguente; la stessa registrazione deve
   contenere 307 immagini RGB e le pose MPS con timestamp device compatibili:

   ```text
   $ARIA/Video_moving_around_room_4_20260928_232041.vrs
   $ARIA/mps_Video_moving_around_room_4_20260928_232041_vrs/
     slam/closed_loop_trajectory.csv
     slam/semidense_points.csv.gz
     hand_tracking/hand_tracking_results.csv
   ```

Il VRS non è sostituibile da un video MP4: contiene calibrazione e timestamp
necessari al matching MPS. La preparazione genera 307 frame RGB 768×576 non
distorti, intrinseci pinhole (~334,48 px) e pose camera MPS in metri.
Per verificare un trasferimento da una macchina all'altra, confrontare gli
SHA-256 di VRS, tre file MPS, checkpoint EgoAllo/ECHO e mesh OMOMO con
`sha256sum` **sulla sorgente e sulla destinazione**; non caricare i file o il
manifest dei dati privati su GitHub.

## 5. Ramo persona: HaMeR → EgoAllo → input ECHO

EgoAllo cerca **esattamente un VRS** sotto `traj_root`, la nuvola MPS sotto
`slam/`, `hamer_outputs.pkl` nella radice e un file chiamato
`wrist_and_palm_poses.csv`. Il nostro MPS Gen2 chiamava quest'ultimo
`hand_tracking_results.csv`: rinominarne una copia; il parser locale
modificato riconosce le colonne Gen2.

```bash
export TRAJ="$EGO/aria_trajectories/room4"
mkdir -p "$TRAJ/slam" "$TRAJ/hand_tracking"
ln -s "$VRS" "$TRAJ/video.vrs"
cp "$MPS/slam/closed_loop_trajectory.csv" "$TRAJ/slam/"
cp "$MPS/slam/semidense_points.csv.gz" "$TRAJ/slam/"
cp "$MPS/hand_tracking/hand_tracking_results.csv" \
  "$TRAJ/hand_tracking/wrist_and_palm_poses.csv"
cd "$EGO"
"$EGO/.mamba-env/bin/python" 2_run_hamer_on_vrs.py --traj-root "$TRAJ"
"$EGO/.mamba-env/bin/python" 3_aria_inference.py --traj-root "$TRAJ" \
  --start-index 30 --traj-length 860 --num-samples 1 \
  --guidance-mode aria_hamer
```

Defaults usati: `guidance_inner=true`, `guidance_post=true`, SMPL-H neutral,
30 Hz. Controllare `hamer_outputs.pkl`,
`egoallo_outputs/<timestamp>_30-890.npz` e il relativo `_args.yaml`.
Quest'ultimo output ha `Ts_world_root`, pose corpo/mani e
`timestamps_ns`; non contiene la traiettoria della sedia. Copiare il path
generato in `EGO_NPZ` per i passi seguenti. Campionamento e ottimizzazione
possono non essere bit-identici su un'altra GPU.

## 6. Ramo sedia: SAM3 → SAM3D → MPS/TAPIR → pose 3D

Partire da una directory output **nuova**, non dal link locale `room4` della
vecchia macchina. Con il nome VRS e i parametri qui sotto, il runner produce
maschere e mesh, poi richiama automaticamente il tracker camera-aware:

```bash
export MICROMAMBA_BIN="$(command -v micromamba)"
export ARIA_ENV_PREFIX="$EGO/.mamba-env"
export TAPIR_GPU=0                   # usare 1 solo se GPU 1 esiste
cd "$RECON"
./run_pipeline_no_hawor.sh "$VRS" 28 whitechair --dry-run
./run_pipeline_no_hawor.sh "$VRS" 28 whitechair
```

Per ripartire da **frame/mask/mesh già preparati**, saltare il primo runner e
usare `./run_camera_aware_room4.sh "$RECON/room4"`; il runner può riutilizzare
gli artefatti intermedi. `STATIC_FRAMES=20:56`, `REFERENCE_FRAME=28` e
`TAPIR_GPU` sono overridabili. Il suo percorso MPS semidense è derivato dal
`vrs` salvato in `room4/aria_metadata.json`, quindi spostare il VRS dopo la
preparazione richiede aggiornare quel metadato o ripreparare l'input.

Output da controllare:

```text
$RECON/room4/aria_camera_trajectory.npz
$RECON/room4/video_segmentation/masks/frame_000028_masks/whitechair/whitechair.obj
$RECON/room4/camera_motion_fix/whitechair_metric.obj
$RECON/room4/camera_motion_fix/mesh_anchor.npz
$RECON/room4/camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz
$RECON/room4/camera_motion_fix/calibrated/metrics.json
```

Sul run storico: 307 slot temporali, **250** pose sedia nel mondo valide;
il tratto continuo poi usato da ECHO è ai frame camera **28–257**. La sedia
grigia viene erroneamente segmentata ai frame 260–272, che il tracking
rifiuta; non interpolare i frame invalidi. La mesh è **una sola** (dal frame
28), calibrata in scala metrica con punti MPS e silhouette. La posa segue
`T_world_object = T_world_camera @ T_camera_object`. Il vecchio
`obj_tracking_out/` Fast-SAM3D è camera-relative, non la traiettoria metrica
da dare a ECHO. Vedere [CAMERA_MOTION_FIX.md](CAMERA_MOTION_FIX.md).

## 7. Allineare OMOMO/SAM e costruire il dataset ECHO

Scaricare il checkpoint ECHO e i tre SMPL-X NPZ nei path della tabella.
Servono il mesh grezzo OMOMO `whitechair_cleaned_simplified.obj` e gli asset
canonici già versionati in ECHO. ECHO è stato addestrato sulla **sedia
OMOMO**: stesso `whitechair` class id 31, keypoint e PointNeXt; la mesh SAM
ha origine, assi e forma diversi. Un cambio di frame rigido non elimina la
differenza di forma. Non applicare direttamente `obj_R/obj_transl` alla mesh
SAM non allineata.

Primo HDF5 (860 frame EgoAllo, oggetto ancora placeholder):

```bash
export EGO_NPZ="$TRAJ/egoallo_outputs/<timestamp>_30-890.npz"
export BASE="$ECHO/data/preprocessed/aria_egoallo_whitechair"
PYTHONPATH="$EGO/src" "$EGO/.mamba-env/bin/python" \
  "$ECHO/scripts/preprocess_egoallo_conditioning.py" \
  --egoallo-output "$EGO_NPZ" --smplh-model "$EGO/data/smplh/neutral/model.npz" \
  --output "$BASE/dataset_test_30fps/room4.hdf5" --echo-root "$ECHO"
```

Questo passaggio canonizza la mesh OMOMO e copia gli asset di contatto. Poi
stimare `T_sam_echo` sui punti MPS e sulle maschere, **senza cambiare la
scala metrica OMOMO**:

```bash
micromamba run -n sam3d python "$RECON/scripts/fit_echo_mesh_to_mps.py" \
  --sequence "$RECON/room4" \
  --landmarks "$RECON/room4/camera_motion_fix/landmarks_triangulated.npz" \
  --sam-anchor "$RECON/room4/camera_motion_fix/mesh_anchor.npz" \
  --echo-mesh "$BASE/object_meshes/whitechair.ply" \
  --output "$RECON/room4/camera_motion_fix/echo_object_alignment.npz"
```

Sul run storico, SAM ha IoU mediana silhouette held-out ~0,851, mentre la
mesh OMOMO allineata rigidamente ~0,483: questa è una limitazione geometrica,
non un errore di path. Costruire il dataset finale sincronizzato (30 Hz,
interpolazione SLERP/traslazione, soglia confidence 0,35):

```bash
export TRACKED="$ECHO/data/preprocessed/aria_egoallo_tracked_whitechair"
mkdir -p "$TRACKED"
cp -a "$BASE/object_keypoints" "$TRACKED/"
cp -a "$BASE/object_meshes" "$TRACKED/"
cp -a "$BASE/object_pointnext.pkl" "$TRACKED/"
python "$ECHO/scripts/build_aria_tracked_object_conditioning.py" \
  --source-hdf5 "$BASE/dataset_test_30fps/room4.hdf5" \
  --egoallo-output "$EGO_NPZ" \
  --object-trajectory "$RECON/room4/camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz" \
  --object-alignment "$RECON/room4/camera_motion_fix/echo_object_alignment.npz" \
  --output "$TRACKED/dataset_test_30fps/room4.hdf5"
```

Qui `python` è quello dell'**ambiente ECHO attivo**. Attesi
`dataset_test_30fps/room4.json` con 693 frame, EgoAllo frame 50–742,
traiettoria sedia sorgente 28–257 e confidence mediana ~0,912. Il builder
seleziona il tratto continuo valido più lungo; cambiando dati/maschere queste
lunghezze possono cambiare. Persona e oggetto devono usare gli **stessi**
timestamp Aria e lo stesso shift del mondo applicato dal preprocess EgoAllo.

## 8. ECHO 017: inferenza e video, poi mesh SAM 018

Da `$ECHO`, con ambiente ECHO attivo e `scenarios/env.yaml` aggiornato:

```bash
python main.py -c scenarios/env.yaml scenarios/echo.yaml \
  scenarios/aria_egoallo_tracked_whitechair.yaml
export SAMPLE017="$ECHO/experiments/017_aria_egoallo_tracked_whitechair/artifacts_egoallo_tracked_whitechair_101/step_-1_samples/aria_test/sbj/repetition_0/room4.hdf5"
python scripts/render_sample.py --input "$SAMPLE017" \
  --tracking "$TRACKED/dataset_test_30fps/room4.hdf5" \
  --object-mesh "$BASE/object_meshes/whitechair.ply" \
  --output "$ECHO/experiments/017_aria_egoallo_tracked_whitechair/room4_egoallo_tracked_whitechair_101.mp4"
```

Il file scenario fissa checkpoint `assets/echo.pth` con EMA, seed 4242,
`sample.modalities="101"` (**corpo e contatti generati, posa oggetto
misurata/fissata**), finestre 60 frame con overlap 30 e 100 step DDPM.
Il run 016, scenario `aria_egoallo_whitechair.yaml`, era una prova `111`
con sedia placeholder: non è la ricostruzione 3D osservata.

Per il video 018, trasformare **i vertici**, non le pose già campionate:
`T_world_echo = T_world_sam @ T_sam_echo`, quindi
`mesh_SAM_in_ECHO = inverse(T_sam_echo) @ mesh_SAM`. Esempio one-shot:

```bash
export SAM_ECHO="$ECHO/experiments/018_aria_egoallo_tracked_whitechair_sam_mesh/whitechair_sam_in_echo_frame.ply"
python - "$RECON/room4/camera_motion_fix/whitechair_metric.obj" \
  "$RECON/room4/camera_motion_fix/echo_object_alignment.npz" "$SAM_ECHO" <<'PY'
import sys
from pathlib import Path
import numpy as np
import trimesh
mesh = trimesh.load(sys.argv[1], force='mesh', process=False)
T_sam_echo = np.load(sys.argv[2], allow_pickle=False)['T_sam_echo']
mesh.apply_transform(np.linalg.inv(T_sam_echo))
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
mesh.export(sys.argv[3])
PY
python scripts/render_sample.py --input "$SAMPLE017" \
  --tracking "$TRACKED/dataset_test_30fps/room4.hdf5" \
  --object-mesh "$SAM_ECHO" \
  --output "$ECHO/experiments/018_aria_egoallo_tracked_whitechair_sam_mesh/room4_egoallo_tracked_whitechair_sam_mesh.mp4"
```

018 **non** è una seconda inferenza ECHO: cambia soltanto la geometria
renderizzata. Il PointNeXt/class ID appreso resta OMOMO.

## 9. Esperimento opzionale 019/020: contatti SAM

Per riprodurre anche il risultato con contatti, dopo 017/018:

```bash
export SAM_CONTACT="$ECHO/data/preprocessed/aria_egoallo_tracked_whitechair_sam_contact"
python scripts/prepare_sam_contact_guidance.py --source "$TRACKED" \
  --sam-mesh-in-echo-frame "$SAM_ECHO" --output "$SAM_CONTACT"
python main.py -c scenarios/env.yaml scenarios/echo.yaml \
  scenarios/aria_egoallo_tracked_whitechair.yaml -- \
  aria.root=./data/preprocessed/aria_egoallo_tracked_whitechair_sam_contact \
  run.name=019_aria_egoallo_sam_contact_guided \
  sample.artifacts_folder_suffix=sam_contact_guided_101 \
  model.cg_scale=0.05 model.cg_contact_threshold=0.025
export SAMPLE019="$ECHO/experiments/019_aria_egoallo_sam_contact_guided/artifacts_sam_contact_guided_101/step_-1_samples/aria_test/sbj/repetition_0/room4.hdf5"
python scripts/optimize_sam_object_contacts.py --input "$SAMPLE019" \
  --mesh "$SAM_ECHO" --conditioning "$TRACKED/dataset_test_30fps/room4.hdf5" \
  --prior-weight 0.5 \
  --output "$ECHO/experiments/020_aria_egoallo_sam_contact_optimized/room4_prior05.hdf5"
python scripts/render_sample.py \
  --input "$ECHO/experiments/020_aria_egoallo_sam_contact_optimized/room4_prior05.hdf5" \
  --tracking "$TRACKED/dataset_test_30fps/room4.hdf5" \
  --object-mesh "$SAM_ECHO" \
  --output "$ECHO/experiments/020_aria_egoallo_sam_contact_optimized/room4_sam_contact_optimized.mp4"
```

`prepare_sam_contact_guidance.py` richiede output **inesistente** e crea
symlink locali: ricreare i link sulla nuova macchina, non copiare quelli
assoluti vecchi. 019 cambia keypoint/mesh, **non** la classe o PointNeXt
OMOMO. 020 ottimizza solo la traslazione dell'oggetto, non il corpo o la
rotazione: correzione massima osservata 3,32 cm; il gap mediano held-out
è sceso 5,66→4,39 cm, ma la coda dei contatti non è migliorata. È un
esperimento geometrico, non una presa fisicamente verificata. Dettagli in
`$ECHO/experiments/020_aria_egoallo_sam_contact_optimized/README.md`
sulla macchina sorgente (quel report non è versionato in ECHO upstream).

## 10. Checklist e problemi che ci hanno fatto perdere tempo

- **Fonte del movimento:** EgoAllo dà testa/polsi/corpo SMPL-H, non la sedia.
  Il tracker MPS dà la sedia. Non usare l'NPZ legacy camera-relative al posto
  di `camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz`.
- **Scala e coordinate:** MPS in metri, `+z` mondo verso l'alto; camera OpenCV
  `x` destra, `y` giù, `z` avanti. Interpolare pose camera sul timestamp
  device, comporre `T_world_camera @ T_camera_object`; mai sommare semplicemente
  traslazioni o assumere che `T_camera_object` sia già nel mondo.
- **Mesh ECHO vs SAM:** la mesh OMOMO è quella vista dal checkpoint ECHO.
  `T_sam_echo` allinea i frame locali, non la forma; 018 è un re-render e 019
  un tentativo di guidare i contatti con keypoint SAM. Nessuna mesh nuova per
  frame e nessun retraining ECHO su SAM.
- **MPS/Gen2:** `hand_tracking_results.csv` va copiato con il nome che EgoAllo
  cerca; il parser upstream 1.6 non legge il nostro formato Gen2. Nessuna
  extrapolazione nei tratti senza MPS o con sedia non osservabile.
- **HF offline/gated:** se manca `sam3.pt` o un `.ckpt/.yaml` SAM3D, il runner
  fallisce subito; se manca la cache MoGe/DINO, può fallire più tardi. Autorizzare
  i modelli gated e pre-caricare i pesi prima del run. `ss_encoder.yaml` del
  fork Fast-SAM3D non è un checkpoint: servono comunque i pesi pesanti.
- **GPU/env:** il default TAPIR è GPU 1: impostare `TAPIR_GPU=0` su host a
  singola GPU. Separare Torch/JAX/ECHO; gli export Conda sono riferimenti,
  non ambienti universali. Una GPU da 20 GiB ha poco margine per SAM3D.
- **Import/compilazione SAM3D:** se `notebook.inference` importa il package
  Jupyter anziché la directory del repository, in `sam3d` disinstallare il
  package `notebook` come in [env/README.md](env/README.md). Su Ada il test
  usava SDPA, non richiedeva `flash_attn`; non installare alla cieca backend
  CUDA compilati per un'altra architettura. I manifest HaWoR contengono anche
  pin Torch/CUDA vecchi, irrilevanti per il percorso Room4 senza HaWoR.
- **Setup SAM3:** il runner passa esplicitamente `SAM3_CHECKPOINT` al builder;
  verificare che il modello gated sia stato scaricato con l'account che esegue
  l'inferenza. La semplice presenza del repository `sam3` non include i pesi.
- **Checkpoint ECHO e SMPL:** `echo.pth` e gli NPZ SMPL non sono nel Git; il
  checkpoint deve corrispondere agli asset OMOMO/PointNeXt del repository.
  Video headless: verificare EGL e `pyrender`.
- **Portabilità:** nei JSON/NPZ possono restare path assoluti verso la vecchia
  macchina (per esempio `fixed_mesh_path`, `aria_metadata.json`); gli script
  che li consumano richiedono path validi. Preferire ricreare gli intermedi
  sulla nuova macchina anziché copiare HDF5/NPZ alla cieca.
- **Confronti attesi:** 860 frame dal preprocess EgoAllo, 693 dopo
  sincronizzazione con sedia, 307 slot camera di cui 250 pose sedia nel mondo
  valide, mesh SAM metrica e video 017/018/020. Divergenze numeriche moderate
  sono possibili; verificare maschere, scala e timestamp prima di attribuirle
  a ECHO.

Per la sola ricostruzione oggetto, leggere anche [NO_HAWOR.md](NO_HAWOR.md),
[CAMERA_MOTION_FIX.md](CAMERA_MOTION_FIX.md) e [ROOM4_READINESS.md](ROOM4_READINESS.md).
Dal root `do-as-i-do`, un test leggero della compensazione del moto camera è:
`micromamba run -n sam3d python -m unittest reconstruction.tests.test_camera_motion_geometry`
(3 test passati sulla macchina sorgente).
