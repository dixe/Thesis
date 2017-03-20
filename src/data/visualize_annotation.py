import img_loader as IML


if __name__ == "__main__":
    img_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/November 6, 2013"

    img_name = "00002-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    imgLoader.visualize_image()
