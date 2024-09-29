import gradio as gr
import video_extract as ext
import os


def load_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


# 假定你有一个名为 "introduction.md" 的 Markdown 文件
changelog_content = load_markdown_file("./CHANGELOG.md")
roadmap_content = load_markdown_file("./ROADMAP.md")


# Gradio界面
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("视频PPT提取应用"):
            with gr.Row():
                gr.Markdown("## 请先上传视频或者指定本地视频文件路径")
            with gr.Row():
                video_input = gr.Video(label="上传视频")
            with gr.Row():
                threshold = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5, label="图片相似度阈值"
                )
                interval_sec = gr.Slider(
                    minimum=0.1, maximum=10, step=0.5, value=1, label="抽帧间隔（秒）"
                )
                merge_size_threshold = gr.Slider(
                    minimum=0, maximum=4096, step=1, value=512, label="字幕合并阈值(B)"
                )
            with gr.Row():
                extract_frame_button = gr.Button("提取并总结ppt")
            with gr.Row():
                gr.Markdown("## Chat With Video")
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox()
                    clear = gr.ClearButton([msg, chatbot])
            with gr.Row():
                gr.Markdown("## 打包下载")
            with gr.Row():
                file_output = gr.File(
                    label="视频总结", show_label=False, file_types=["zip"]
                )

            msg.submit(ext.chat_respond, [msg, chatbot], [msg, chatbot])
            # 添加示例
            examples = gr.Examples(
                examples=[
                    ["/videos/example_video1.mp4", 5, 1.5, 128],
                    ["/videos/example_video2.mp4", 5, 1.5, 128],
                ],
                inputs=[video_input, threshold, interval_sec, merge_size_threshold],
            )
            extract_frame_button.click(
                fn=ext.extract_and_summary_video,
                inputs=[video_input, interval_sec, threshold, merge_size_threshold],
                outputs=[chatbot, file_output],
            )
        with gr.TabItem("应用介绍"):
            with gr.Row():
                gr.Markdown("## 技术原理介绍")
            with gr.Column():
                # 定义一个函数来提取文件名中的数字
                def extract_number(filename):
                    return int("".join(filter(str.isdigit, filename)))

                # 获取图片文件列表并按数字排序
                imgs_dir = "imgs"
                img_files = sorted(
                    [
                        f
                        for f in os.listdir(imgs_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                    ],
                    key=extract_number,
                )

                for img_file in img_files:
                    img_path = os.path.join(imgs_dir, img_file)
                    gr.Image(img_path, label=img_file)

        with gr.TabItem("CHANGELOG"):
            with gr.Row():
                gr.Markdown(changelog_content)
        with gr.TabItem("ROADMAP"):
            with gr.Row():
                gr.Markdown(roadmap_content)
        with gr.TabItem("issue"):
                gr.HTML('<a href="https://github.com/king-jingxiang/super-rag/issues" target="_blank">issues</a>')

demo.launch(server_name="0.0.0.0")
