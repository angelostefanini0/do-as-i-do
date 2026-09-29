# Verifica room4 / white chair — 29 settembre 2026

Pronti ad avviare `run_pipeline_no_hawor.sh` per la ricostruzione dell'oggetto.
Le dipendenze runtime hanno superato prove reali sulla sequenza richiesta,
compresi mesh con texture, tracking bidirezionale e NPZ finale. La ricostruzione
completa dei 307 frame non è stata eseguita.

Comando da eseguire dalla directory `reconstruction`:

```bash
./run_pipeline_no_hawor.sh \
  /home/astefanini/AriaData/Video_moving_around_room_4_20260928_232041.vrs \
  28 whitechair
```

Il runner usa il prompt testuale `white chair`, senza GUI, e scrive in `room4/`.
Questa directory non è stata creata dalle prove. Per una nuova esecuzione,
se esistono già frame estratti, usare una directory vuota con `--output-dir`.
HaWoR, MANO e l'ottimizzazione basata sulla mano non sono richiesti da questo runner.

## Risultati

| Componente | Verifica | Esito |
|---|---|---|
| Runner | Sintassi Bash e dry-run dell'intera sequenza di comandi | OK |
| Project Aria Tools | Estrazione e undistortion di tutti i 307 frame RGB | OK |
| Input | Immagini 768 × 576; timestamp crescenti; durata 30,5968 s | OK |
| MPS | Allineamento dei pose camera ai timestamp RGB | 294/307 validi; frame 28 valido |
| FFmpeg | Codifica lossless e decodifica del video di prova | OK |
| SAM3 | Prompt `white chair` e propagazione delle maschere sui frame originali 27–29 | OK |
| SAM3D | Caricamento checkpoint, mesh della sedia al frame 28, postprocess e texture baking | OK |
| MoGe | Inferenza pointmap e intrinsics sul frame 28 | OK |
| GeoCalib | Inferenza gravity su un frame, modalità camera mobile | OK |
| BootsTAPIR | Caricamento checkpoint e motion stats su due transizioni | OK |
| Fast-SAM3D | Tracking sui frame originali 27–29, ancorato al frame 28 | OK |
| Conversione camera | Conversione del layout prodotto dal tracker | OK |
| Esportazione | Mesh a scala fissa e NPZ con 3/3 pose rigide finite, rotazioni valide e timestamp corrispondenti | OK |
| Kernel CUDA | Rasterizzazione PyTorch3D e nvdiffrast; convoluzione spconv | OK |
| Rasterizzatore Gaussian | Libreria compilata caricata e usata dal texture baking | OK |

Il tracking è stato provato con i parametri del runner: 25 pose samples,
25 Euler steps, batch chunk predefinito 8, SS cache, `torch.compile`,
chaining, render-IoU/cluster, scala fissa e post-optimization senza shape ICP.
È stato eseguito anche il percorso predefinito di decoder inversion da 1000 step:
l'encoder SS opzionale non è configurato e non è necessario.

## Ambienti e checkpoint

| Ambiente | Python | Torch |
|---|---|---|
| `sam3` | 3.12.0 | 2.8.0+cu128 |
| `sam3d` | 3.11.0 | 2.8.0+cu128 |
| `tapnet` | 3.10.20 | 2.7.0+cu128 |
| Aria | `/home/astefanini/egoallo/.mamba-env` | Project Aria Tools disponibili |

I checkpoint SAM3, SAM3D, MoGe, GeoCalib e BootsTAPIR erano già presenti.
È stata completata la cache Torch Hub mancante di `facebookresearch/dinov2`
e `dinov2_vitl14_reg4_pretrain.pth` (circa 1,2 GB). Dopo questo caricamento,
mesh e tracking sono stati provati con Hugging Face offline e accesso URL
Torch Hub bloccato durante la prova: non hanno richiesto altri download.
Su queste GPU Ada viene usato il backend SDPA; `flash_attn` non è richiesto.

`pip check` è pulito in `tapnet`. In `sam3` segnala il tag di piattaforma di
`decord`, sebbene la segmentazione video reale funzioni. In `sam3d` segnala
molte dipendenze/pin del manifesto generale di SAM3D, inclusi strumenti di
sviluppo, training e altri backend, oltre a un disallineamento Jupyter.
Questi ambienti non sono certificati come installazioni di tutte le dipendenze
dichiarate dal pacchetto; il percorso runtime oggetto senza HaWoR è stato
verificato tramite le inferenze sopra. Non sono stati installati pacchetti
Python aggiuntivi né cambiati i pin degli ambienti.

## Limiti della verifica

La macchina ha tre RTX 4000 Ada da circa 20 GiB ciascuna; il runner usa una
singola GPU, selezionata come GPU 0. Mesh e tracking hanno completato le prove
su GPU 0 con picchi PyTorch di memoria allocata rispettivamente di 19,70 GB
e 16,78 GB (valori decimali). Il README generale richiede almeno 32 GB:
il test breve riesce su questa macchina, ma il margine della mesh è ridotto
e non certifica il consumo della segmentazione su tutti i 307 frame.

Il percorso predefinito produce pose relative alla camera in unità
SAM3D/MoGe non verificate metricamente. Non applica compensazione esplicita
del moto camera al chaining senza `--hdf5-path`; le pose MPS sono esportate
separatamente. La composizione oggetto nel mondo richiede una scala verificata.
I frame senza MPS sono 0–11 e 306, mantenuti invalidi senza extrapolazione.
La verifica riguarda l'eseguibilità, non l'accuratezza del tracking su tutta
la movimentazione della sedia.

Log e artefatti della prova sono salvati in
[`weights/readiness_room4_20260929/`](weights/readiness_room4_20260929/),
esclusi da Git. La sottosequenza di prova rinumera 27, 28, 29 come 0, 1, 2,
preservando i timestamp Aria originali. Include `mesh-smoke.log`,
`tracker-smoke.log`, `export-smoke.log` e
`smoke/room4_whitechair_trajectory.npz`.
