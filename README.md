# extract_video_ppt_to_markdown

#### 介绍

* 视频内容提取： 自动提取视频中的PPT图片，帮助学习者快速定位关键视觉信息。
* 音频转字幕： 使用ASR模型将视频音频转换为字幕，增强内容的可访问性。
* 字幕总结： 利用大语言模型对字幕进行总结，生成结构化的Markdown文件，便于复习和资料整理。
* 基于知识库的问答系统： 根据视频提取的字幕和总结构建基于知识库的问答系统
* 提取内容导出： 导出提取的视频帧和总结行程markdown课件

### 技术挑战

- Token上下文限制： 大型语言模型如Qwen2-7B-Instruct在处理长文本时会受到token上下文限制，导致无法处理过长的视频字幕片段。这需要通过分段处理、上下文拼接等技术手段来解决。
- 显存容量限制： 处理大型模型时，显存容量可能成为瓶颈，特别是在多任务并发处理时。需要通过模型压缩、量化、分布式计算等技术来优化显存使用。
- 推理延迟： 当前推理延迟较高，不适合在线实时处理。需要通过模型优化、硬件加速、并行计算等手段来降低延迟。
- 模型泛化能力： 模型需要适应不同类型和质量的视频内容，这要求模型具有较强的泛化能力和鲁棒性。
- 网络延迟：网络延迟较高，稳定性查，在测试过程中没有本地测试更便捷

### 使用的模型：
- ASR模型： OpenAI/Whisper-large-v3
- LLM模型： Qwen/Qwen2-7B-Instruct
- Embedding模型： TencentBAC/Conan-embedding-v1

### 应用介绍
![](./imgs/视频课件提取Agent应用介绍-图片-1.jpg)

![](./imgs/视频课件提取Agent应用介绍-图片-2.jpg)

![](./imgs/视频课件提取Agent应用介绍-图片-3.jpg)

![](./imgs/视频课件提取Agent应用介绍-图片-4.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-5.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-6.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-7.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-8.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-9.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-10.jpg)
![](./imgs/视频课件提取Agent应用介绍-图片-11.jpg)