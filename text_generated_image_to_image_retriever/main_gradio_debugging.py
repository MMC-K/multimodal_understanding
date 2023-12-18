import gradio as gr
import utils
from PIL import Image
from ko_to_en_translation import translate_ko2en #, google_translate_ko2en
from text_to_image_generation import generate_image, generate_controlnet_image
from image_to_image_retrieval import retrieve_image_with_image, retrieve_image_with_text, retrieve_image_with_multiple_images

# from text_to_image_retrieval import retrieve_image_with_text

# def get_html(result, caption, img_size=180):
#     result_html = f"""
# <table>
# <caption>{caption}</caption>
# <tr>
#     <td>Top 1 ({result[0]['score']:.2f})</td>
#     <td>Top 2 ({result[1]['score']:.2f})</td>
#     <td>Top 3 ({result[2]['score']:.2f})</td>
#     <td>Top 4 ({result[3]['score']:.2f})</td>
# </tr>
# <tr>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[0]['image_url']}"
#             alt="score: {result[0]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[1]['image_url']}"
#             alt="score: {result[1]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[2]['image_url']}"
#             alt="score: {result[2]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[3]['image_url']}"
#             alt="score: {result[3]['score']:.2f}">
#     </td>
# </tr>
# <tr>
#     <td>Top 5 ({result[4]['score']:.2f})</td>
#     <td>Top 6 ({result[5]['score']:.2f})</td>
#     <td>Top 7 ({result[6]['score']:.2f})</td>
#     <td>Top 8 ({result[7]['score']:.2f})</td>
# </tr>
# <tr>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[4]['image_url']}"
#             alt="score: {result[4]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[5]['image_url']}"
#             alt="score: {result[5]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[6]['image_url']}"
#             alt="score: {result[6]['score']:.2f}">
#     </td>
#     <td>
#         <img height="{img_size}" width="{img_size}"
#             src="{result[7]['image_url']}"
#             alt="score: {result[7]['score']:.2f}">
#     </td>
# </tr>
# </table>
#             """
#     return result_html
# def retrieve(prompt):
#     english_prompts = translate_ko2en([prompt])
#     generated_images = generate_image(english_prompts)
#     result_with_image = retrieve_image_with_image(generated_images)[0]
#     result_with_text = retrieve_image_with_text([prompt])[0]
#     # result_html = "\n".join([f"<img src='{ith['image_url']}' alt={ith['score']} />" for ith in result])
    
#     result_with_image_html = get_html(result_with_image, caption="text-image-image") 
#     result_with_text_html = get_html(result_with_text, caption="text-image")
    
#     print("[*] result_with_image:", result_with_image)
#     print("[*] result_with_text:", result_with_text)

#     return english_prompts[0], generated_images[0], result_with_image_html, result_with_text_html

# demo = gr.Interface(fn=retrieve, inputs="text", outputs= ["text", gr.Image(type="pil"), "html", "html"])
    
# if __name__ == "__main__":
#     demo.launch(show_api=False)   

# def increase(num):
#     return num + 1

def translate(korean_text):
    return translate_ko2en([korean_text])[0]

generated_images = None
def generate(english_text, common_english_positive_text, common_english_negative_text, gs, pose_image, batch_size_number):
    # return generate_image([english_text])[0]
    global generated_images
    generated_images = generate_controlnet_image([english_text], common_positive_prompts=common_english_positive_text, common_negative_prompts=common_english_negative_text, guidance_scale=gs, pose_image=pose_image, batch_size=batch_size_number)[0]
    print(generated_images)
    dst = Image.new('RGB', (sum([gi.width for gi in generated_images]), generated_images[0].height))
    x = 0
    for gi in generated_images:
        dst.paste(gi, (x, 0))
        x += gi.width
        
    return dst

def generate_wo_pose(english_text, common_english_positive_text, common_english_negative_text, gs, batch_size_number):
    # return generate_image([english_text])[0]
    global generated_images
    generated_images = generate_image([english_text], common_positive_prompts=common_english_positive_text, common_negative_prompts=common_english_negative_text, guidance_scale=gs, batch_size=batch_size_number)[0]
    print(generated_images)
    dst = Image.new('RGB', (sum([gi.width for gi in generated_images]), generated_images[0].height))
    x = 0
    for gi in generated_images:
        dst.paste(gi, (x, 0))
        x += gi.width
        
    return dst 

def get_html(result, caption, img_size=180):
    result_html = f"""
<table>
<caption>{caption}</caption>
<tr>
    <td>Top 1 ({result[0]['score']:.2f})</td>
    <td>Top 2 ({result[1]['score']:.2f})</td>
    <td>Top 3 ({result[2]['score']:.2f})</td>
    <td>Top 4 ({result[3]['score']:.2f})</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[0]['image_url']}"
            alt="score: {result[0]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[1]['image_url']}"
            alt="score: {result[1]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[2]['image_url']}"
            alt="score: {result[2]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[3]['image_url']}"
            alt="score: {result[3]['score']:.2f}">
    </td>
</tr>

<tr>
    <td>
        {result[0]['tags']}
    </td>
    <td>
         {result[1]['tags']}
    </td>
    <td>
         {result[2]['tags']}
    </td>
    <td>
         {result[3]['tags']}
    </td>
</tr>
</table>
            """
    return result_html

# def convert_to_html():

def retrieve(text_query):
    global generated_images
    result_with_image = retrieve_image_with_multiple_images([generated_images])[0]
    result_with_text = retrieve_image_with_text([text_query])[0]

    for r in result_with_image:
        img_id = r["image_url"].split("/")[-1].split(".")[-2]
        r["tags"] = utils.id_to_tags(img_id)

    for r in result_with_text:
        img_id = r["image_url"].split("/")[-1].split(".")[-2]
        r["tags"] = utils.id_to_tags(img_id)      



    html1 = get_html(result_with_image, caption="image generative")
    html2 = get_html(result_with_text, caption="text")

    return html1, html2

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            korean_text = gr.Text(label="Korean")
            translate_button = gr.Button("Translate")


            english_positive_text = gr.Text(label="English postive")
            common_english_positive_text = gr.Text(label="English common postive", value="In a photo studio, a female 35-year-old fashion model wearing, best quality")
            common_english_negative_text = gr.Text(label="English common negative", value="looking back, cartoon, anime, mannequin, illustration, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, stage equipment")
            
            with gr.Row():
                guidance_scale_number = gr.Number(label="guidance_scale", value=13)
                batch_size_number = gr.Number(label="batch_size_number", value=1, precision=0)
            
            pose_image = gr.Image(label="Pose Image", interactive=True, show_download_button=True, image_mode="RGBA", type="pil")
            
            with gr.Row():
                #with gr.Column(scale=1):
                generate_button = gr.Button("Generate w/ pose")
                #with gr.Column(scale=1):
                generate_wo_pose_button = gr.Button("Generate w/o pose")


            generated_image = gr.Image(label="Generated Image", interactive=True, show_download_button=True, sources=["upload", "webcam", "clipboard"])
            retrieve_button = gr.Button("Retrieve!")
        
        with gr.Column(scale=4):
            out_html_1 = gr.HTML("Generative retrieval")
            out_html_2 = gr.HTML("text retrieval")
        
        generate_button.click(generate, [english_positive_text, common_english_positive_text, common_english_negative_text, guidance_scale_number, pose_image, batch_size_number], generated_image)
        generate_wo_pose_button.click(generate_wo_pose, [english_positive_text, common_english_positive_text, common_english_negative_text, guidance_scale_number, batch_size_number], generated_image)
        translate_button.click(translate, korean_text, english_positive_text)
        retrieve_button.click(retrieve, inputs=korean_text, outputs=[out_html_1, out_html_2])

    # btoa.click(increase, b, a)

demo.launch()