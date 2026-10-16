# Phase 2 status

## 1. Data
[OK]   train: 4040 pairs (expected 4040)
[OK]   CHAMELEON: 76 pairs (expected 76)
[OK]   CAMO: 250 pairs (expected 250)
[OK]   COD10K: 2026 pairs (expected 2026)
[OK]   NC4K: 4121 pairs (expected 4121)

## 2. Weights
[OK]   SINet-V2: checkpoints\sinet_v2\Net_epoch_best.pth (103 MB, 880 tensors)
[OK]   PFNet: checkpoints\pfnet\PFNet.pth (178 MB, 965 tensors)
[OK]   ZoomNet: checkpoints\zoomnet\cod_zoomnet_r50_bs8_e40_2022-03-04.pth (124 MB, 901 tensors)
[INFO] FEDER: no weights (official link dead) -> clean-only from released maps, checked in §4/§5
[OK]   SAM2-UNet: checkpoints\sam2_unet\Pretrained checkpoints\SAM2UNet-COD.pth (826 MB, 1192 tensors)
[OK]   SAM-Adapter: checkpoints\sam_adapter\model_epoch_best.pth (2446 MB, 649 tensors)  
[OK]   Oracle-Box-MobileSAM: checkpoints\mobile_sam\mobile_sam.pt (39 MB, 439 tensors)   

## 3. SAM-Adapter released config vs our wrapper
[OK]   config matches the wrapper (embed_dim 1280, depth 32, inp_size 1024)

## 4. Released prediction maps
[OK]   SINet-V2 maps CHAMELEON: 76/76 PNGs
[OK]   SINet-V2 maps CAMO: 250/250 PNGs
[OK]   SINet-V2 maps COD10K: 2026/2026 PNGs
[OK]   SINet-V2 maps NC4K: 4121/4121 PNGs
[OK]   ZoomNet maps CHAMELEON: 76/76 PNGs
[OK]   ZoomNet maps CAMO: 250/250 PNGs
[OK]   ZoomNet maps COD10K: 2026/2026 PNGs
[OK]   ZoomNet maps NC4K: 4121/4121 PNGs
[OK]   FEDER maps CHAMELEON: 76/76 PNGs
[OK]   FEDER maps CAMO: 250/250 PNGs
[TODO] FEDER maps: no COD10K folder under data/released_maps/FEDER
[OK]   FEDER maps NC4K: 4121/4121 PNGs

## 5. Clean results
[OK]   SINet-V2              CHAMELEON Sm=0.8881 (fp32)
[OK]   SINet-V2              CAMO      Sm=0.8204 (fp32)
[OK]   SINet-V2              COD10K    Sm=0.8152 (fp32)
[OK]   SINet-V2              NC4K      Sm=0.8470 (fp32)
[OK]   PFNet                 CHAMELEON Sm=0.8819 (fp32)
[OK]   PFNet                 CAMO      Sm=0.7825 (fp32)
[OK]   PFNet                 COD10K    Sm=0.7996 (fp32)
[OK]   PFNet                 NC4K      Sm=0.8287 (fp32)
[OK]   ZoomNet               CHAMELEON Sm=0.9019 (fp32)
[OK]   ZoomNet               CAMO      Sm=0.8201 (fp32)
[OK]   ZoomNet               COD10K    Sm=0.8385 (fp32)
[OK]   ZoomNet               NC4K      Sm=0.8531 (fp32)
[OK]   SAM2-UNet             CHAMELEON Sm=0.9139 (fp16)
[OK]   SAM2-UNet             CAMO      Sm=0.8835 (fp16)
[OK]   SAM2-UNet             COD10K    Sm=0.8799 (fp16)
[OK]   SAM2-UNet             NC4K      Sm=0.9007 (fp16)
[OK]   SAM-Adapter           CHAMELEON Sm=0.8959 (fp16)
[OK]   SAM-Adapter           CAMO      Sm=0.8470 (fp16)
[OK]   SAM-Adapter           COD10K    Sm=0.8844 (fp16)
[OK]   SAM-Adapter           NC4K      Sm=0.8842 (fp16)
[OK]   Oracle-Box-MobileSAM  CHAMELEON Sm=0.7869 (fp32)
[OK]   Oracle-Box-MobileSAM  CAMO      Sm=0.8270 (fp32)
[OK]   Oracle-Box-MobileSAM  COD10K    Sm=0.8273 (fp32)
[OK]   Oracle-Box-MobileSAM  NC4K      Sm=0.8503 (fp32)
[OK]   FEDER                 CHAMELEON Sm=0.8867 (clean-only, released maps, 0 missing)  
[OK]   FEDER                 CAMO      Sm=0.8021 (clean-only, released maps, 0 missing)  
[TODO] FEDER COD10K: no clean-only result
[OK]   FEDER                 NC4K      Sm=0.8470 (clean-only, released maps, 0 missing)  
[TODO] SINet-V2: released maps not scored

## 6. Reproduction
[OK]   SINet-V2              CHAMELEON ours 0.8881 vs paper 0.888 (Δ +0.0001)
[OK]   SINet-V2              CAMO      ours 0.8204 vs paper 0.820 (Δ +0.0004)
[OK]   SINet-V2              COD10K    ours 0.8152 vs paper 0.815 (Δ +0.0002)
[OK]   SINet-V2              NC4K      ours 0.8470 vs paper 0.847 (Δ -0.0000)
[OK]   ZoomNet               CHAMELEON ours 0.9019 vs paper 0.902 (Δ -0.0001)
[OK]   ZoomNet               CHAMELEON ours vs authors' maps Δ +0.0002
[OK]   ZoomNet               CAMO      ours 0.8201 vs paper 0.820 (Δ +0.0001)
[OK]   ZoomNet               COD10K    ours 0.8385 vs paper 0.838 (Δ +0.0005)
[INFO] ZoomNet               NC4K      ours 0.8531: no confirmed published value
[OK]   ZoomNet               NC4K      ours vs authors' maps Δ +0.0003
[OK]   SAM2-UNet             CHAMELEON ours 0.9139 vs paper 0.914 (Δ -0.0001)
[OK]   SAM2-UNet             CAMO      ours 0.8835 vs paper 0.884 (Δ -0.0005)
[OK]   SAM2-UNet             COD10K    ours 0.8799 vs paper 0.880 (Δ -0.0001)
[OK]   SAM2-UNet             NC4K      ours 0.9007 vs paper 0.901 (Δ -0.0003)
[OK]   PFNet                 CHAMELEON ours 0.8819 vs paper 0.882 (Δ -0.0001)
[OK]   PFNet                 CAMO      ours 0.7825 vs paper 0.782 (Δ +0.0005)
[OK]   PFNet                 COD10K    ours 0.7996 vs paper 0.800 (Δ -0.0004)
[INFO] PFNet                 NC4K      ours 0.8287: no confirmed published value
[OK]   SAM-Adapter           CHAMELEON ours 0.8959 vs paper 0.896 (Δ -0.0001)
[OK]   SAM-Adapter           CAMO      ours 0.8470 vs paper 0.847 (Δ +0.0000)
[OK]   SAM-Adapter           COD10K    ours 0.8844 vs paper 0.883 (Δ +0.0014)
[INFO] SAM-Adapter           NC4K      ours 0.8842: no confirmed published value
[INFO] Oracle-Box-MobileSAM has no published value; inspect a few predictions at Checkpoint 2 (python -m src.evaluate --models Oracle-Box-MobileSAM --datasets CHAMELEON --save-pred 12 --out results/oracle_check.csv)

## Verdict
NOT DONE - work remaining (see [TODO] lines)

## Still to run
    python scripts/eval_prediction_maps.py --model FEDER --pred-root data/released_maps/FEDER
    python scripts/eval_prediction_maps.py --model SINet-V2 --pred-root data/released_maps/SINet-V2
    python scripts/make_reproduction.py
    python scripts/check_phase2.py