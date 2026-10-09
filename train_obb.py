"""Treino do YOLO11n-OBB para detectar a lipobag e sua orientação.

Uso:
    pip install ultralytics albumentations
    python train_obb.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "lipobag.yaml"

# 640 enxerga melhor a bag pequena (drone alto). Se o FPS na Raspberry Pi 5
# ficar baixo, retreine com 480: o modelo deve rodar no mesmo tamanho do treino.
IMGSZ = 640
EPOCHS = 150


def blur_and_noise():
    """Borrão de movimento, desfoque e ruído de sensor (vibração do drone).

    São transformações só de pixel, então não mexem nas caixas rotacionadas.
    """
    try:
        import albumentations as A
    except ImportError:
        print("AVISO: albumentations não instalado, treinando sem blur/ruído.")
        return None
    return [
        A.MotionBlur(blur_limit=9, p=0.25),
        A.GaussianBlur(blur_limit=(3, 7), p=0.10),
        A.GaussNoise(p=0.15),
    ]


def main():
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        print("AVISO: GPU não encontrada, o treino na CPU será bem lento.")

    model = YOLO("yolo11n-obb.pt")  # pesos pré-treinados (transfer learning)

    model.train(
        data=str(DATA),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=30,  # para cedo se a validação não melhorar em 30 épocas
        batch=-1 if has_gpu else 8,  # -1 = ajusta sozinho à memória da GPU
        device=0 if has_gpu else "cpu",
        workers=8 if has_gpu else 2,
        cos_lr=True,
        seed=0,
        project=str(ROOT / "runs"),
        name="lipobag_obb",
        # --- Geometria ---
        degrees=180,  # a bag pode estar em qualquer ângulo
        flipud=0.5,  # câmera olha para baixo: não existe "cima" na imagem
        fliplr=0.5,
        scale=0.7,  # altura do drone varia bastante (bag grande e pequena)
        translate=0.2,  # bag fora do centro e parcialmente cortada na borda
        perspective=0.0005,  # câmera inclina junto com o drone
        shear=2.0,
        # --- Cor (indoor: iluminação varia pouco) ---
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        # --- Composição ---
        mosaic=1.0,  # varia fundo e escala; ajuda muito com dataset pequeno
        close_mosaic=15,  # últimas épocas sem mosaic, só com imagens reais
        mixup=0.0,  # sobrepor imagens atrapalha a borda/ângulo da caixa
        # --- Blur e ruído ---
        augmentations=blur_and_noise(),
    )

    # Valida o melhor checkpoint e exporta para a Raspberry Pi 5 (NCNN).
    best = YOLO(model.trainer.best)
    metrics = best.val(data=str(DATA), imgsz=IMGSZ)
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    best.export(format="ncnn", imgsz=IMGSZ)


if __name__ == "__main__":
    main()
