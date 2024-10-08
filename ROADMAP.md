# Roadmap

## 模型能力提升

1. **验证图像和音频的embedding实现**
   - 开发阶段：验证并实现图像和音频的embedding算法。
   - 目标：提高图像和音频内容的理解和检索能力。

2. **支持通过图像特征提取技术将图像进行向量存储**
   - 开发阶段：实现图像特征提取技术，并将其转化为向量存储。
   - 目标：使图像内容更容易检索和分析。

3. **支持通过音频特征提取技术将音频进行向量存储**
   - 开发阶段：实现音频特征提取技术，并将其转化为向量存储。
   - 目标：提高音频内容的检索效率。

4. **支持通过LVLM (Large Vision Language Model) 将图像转为文本，然后进行向量存储**
   - 开发阶段：集成LVLM模型，实现图像到文本的转换，并存储为向量。
   - 目标：使图像内容可以通过文本搜索。

5. **支持通过LALM (Large Audio Language Model) 将音频转为文本，然后进行向量存储**
   - 开发阶段：集成LALM模型，实现音频到文本的转换，并存储为向量。
   - 目标：提高音频内容的可读性和检索能力。

6. **实现基于图像/音频特征的向量检索**
   - 开发阶段：开发向量检索引擎，支持基于图像和音频特征的检索。
   - 目标：提供高效的内容检索功能。

7. **构建一个多模态的向量存储系统**
   - 开发阶段：整合图像、音频和文本的向量存储系统。
   - 目标：建立统一的多模态数据存储框架。

8. **支持多模态数据检索**
   - 开发阶段：实现跨模态的数据检索功能。
   - 目标：提供综合性的检索体验。

9. **引入Multimodal model**
   - 开发阶段：集成多模态模型，实现多模态内容的理解和处理。
   - 目标：增强系统对复杂场景的理解能力。

10. **支持多模态问题总结**
    - 开发阶段：实现基于多模态内容的问题总结功能。
    - 目标：生成综合性的内容摘要。

11. **支持单卡推理，模型动态加载和卸载**
    - 开发阶段：优化模型加载机制，支持单卡推理。
    - 目标：提高资源利用率，减少延迟。

12. **支持多卡推理，模型常驻**
    - 开发阶段：优化模型部署机制，支持多卡推理。
    - 目标：提高大规模数据处理的效率。

## 交互页面改进

1. **支持通过API的方式自定义配置LLM/Embedding/ASR/LVLM/LALM/Multimodal模型**
    - 开发阶段：开发相应的API接口，允许用户自定义配置模型。
    - 目标：提供灵活的模型配置选项。

2. **浏览器插件的支持**
    - 开发阶段：开发浏览器插件，支持在浏览器中直接使用ChatWithVideo的功能。
    - 目标：方便用户随时随地使用项目功能。

3. **桌面客户端软件的支持**
    - 开发阶段：开发桌面客户端应用程序，提供离线使用功能。
    - 目标：满足用户在不同场景下的使用需求。