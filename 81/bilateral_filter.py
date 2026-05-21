import cv2
import numpy as np


def bilateral_filter(image, sigma_space, sigma_color):
    if image is None:
        raise ValueError("输入图像不能为空")
    
    if len(image.shape) == 2:
        filtered = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_filtered = cv2.bilateralFilter(l, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    lab_filtered = cv2.merge((l_filtered, a, b))
    filtered = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)
    
    return filtered


def joint_bilateral_filter(image, guidance, sigma_space, sigma_color):
    if image is None or guidance is None:
        raise ValueError("输入图像和引导图像都不能为空")
    
    if image.shape[:2] != guidance.shape[:2]:
        raise ValueError("输入图像和引导图像的尺寸必须一致")
    
    if len(image.shape) == 2:
        guidance_gray = guidance if len(guidance.shape) == 2 else cv2.cvtColor(guidance, cv2.COLOR_BGR2GRAY)
        filtered = cv2.ximgproc.jointBilateralFilter(guidance_gray, image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    guidance_lab = cv2.cvtColor(guidance, cv2.COLOR_BGR2LAB)
    guidance_l, _, _ = cv2.split(guidance_lab)
    
    l_filtered = cv2.ximgproc.jointBilateralFilter(guidance_l, l, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    lab_filtered = cv2.merge((l_filtered, a, b))
    filtered = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)
    
    return filtered


def joint_bilateral_filter_manual(image, guidance, sigma_space, sigma_color):
    if image is None or guidance is None:
        raise ValueError("输入图像和引导图像都不能为空")
    
    if image.shape[:2] != guidance.shape[:2]:
        raise ValueError("输入图像和引导图像的尺寸必须一致")
    
    is_gray = len(image.shape) == 2
    
    if is_gray:
        img = image.astype(np.float32)
        guide = guidance.astype(np.float32) if len(guidance.shape) == 2 else cv2.cvtColor(guidance, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        img = lab[:, :, 0]
        guide_lab = cv2.cvtColor(guidance, cv2.COLOR_BGR2LAB).astype(np.float32)
        guide = guide_lab[:, :, 0]
    
    height, width = img.shape
    result = np.zeros_like(img)
    
    radius = int(3 * sigma_space)
    
    for y in range(height):
        for x in range(width):
            y_min = max(0, y - radius)
            y_max = min(height, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(width, x + radius + 1)
            
            img_patch = img[y_min:y_max, x_min:x_max]
            guide_patch = guide[y_min:y_max, x_min:x_max]
            
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            spatial_dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
            spatial_weight = np.exp(-spatial_dist ** 2 / (2 * sigma_space ** 2))
            
            range_dist = np.abs(guide_patch - guide[y, x])
            range_weight = np.exp(-range_dist ** 2 / (2 * sigma_color ** 2))
            
            weight = spatial_weight * range_weight
            weight_sum = np.sum(weight)
            
            if weight_sum > 1e-6:
                result[y, x] = np.sum(img_patch * weight) / weight_sum
            else:
                result[y, x] = img[y, x]
    
    if is_gray:
        return result.astype(image.dtype)
    else:
        lab[:, :, 0] = result
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def main():
    image_path = "test_image.jpg"
    guidance_path = "guidance_image.jpg"
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    sigma_space = 50
    sigma_color = 50
    
    result_bilateral = bilateral_filter(image, sigma_space, sigma_color)
    cv2.imwrite("bilateral_result.jpg", result_bilateral)
    print("双边滤波完成，结果已保存到 bilateral_result.jpg")
    
    guidance = cv2.imread(guidance_path)
    if guidance is not None:
        try:
            result_joint = joint_bilateral_filter(image, guidance, sigma_space, sigma_color)
            cv2.imwrite("joint_result.jpg", result_joint)
            print("联合双边滤波完成，结果已保存到 joint_result.jpg")
        except AttributeError:
            print("未安装 opencv-contrib，使用手动实现的联合双边滤波...")
            result_joint = joint_bilateral_filter_manual(image, guidance, sigma_space, sigma_color)
            cv2.imwrite("joint_result.jpg", result_joint)
            print("联合双边滤波完成，结果已保存到 joint_result.jpg")
    else:
        print(f"未找到引导图像 {guidance_path}，跳过联合双边滤波")


if __name__ == "__main__":
    main()
