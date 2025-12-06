from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import sys
import os

# 添加项目路径
sys.path.append('.')

app = Flask(__name__)

# 加载模型
def load_model():
    try:
        # 导入你的模型
        from model import Model  # 根据你的实际代码调整
        
        # 初始化模型
        model = Model(
            # 你的模型参数
        )
        
        # 加载权重
        checkpoint = torch.load('model.pth', map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("模型加载成功！")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

model = load_model()

# 图片预处理
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),  # 根据你的模型调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image)
    return image.unsqueeze(0)  # 添加batch维度

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/recognize', methods=['POST'])
def recognize_formula():
    try:
        # 获取图片
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({"error": "请提供图片文件或base64"}), 400
        
        if 'image' in request.files:
            image_file = request.files['image']
            image_bytes = image_file.read()
        else:
            # base64格式
            image_base64 = request.json['image_base64']
            image_bytes = base64.b64decode(image_base64)
        
        # 预处理
        input_tensor = preprocess_image(image_bytes)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            # 根据你的模型输出格式调整
            result = decode_output(output)  # 你需要实现这个函数
        
        return jsonify({
            "success": True,
            "latex": result,
            "message": "识别成功"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def decode_output(output):
    """
    将模型输出转换为LaTeX字符串
    根据你的实际模型调整
    """
    # 示例：假设你的模型输出是token序列
    # 你需要根据实际情况实现
    idx2word = {}  # 加载你的词表
    tokens = output.argmax(dim=-1).squeeze().tolist()
    formula = ' '.join([idx2word.get(idx, '') for idx in tokens if idx != 0])
    return formula

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)