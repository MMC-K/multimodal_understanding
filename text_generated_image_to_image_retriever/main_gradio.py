import gradio as gr
from ko_to_en_translation import translate_ko2en 
from text_to_image_generation import generate_image 
from image_to_image_retrieval import retrieve_image_with_image, retrieve_image_with_text
# from text_to_image_retrieval import retrieve_image_with_text
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
    <td>Top 5 ({result[4]['score']:.2f})</td>
    <td>Top 6 ({result[5]['score']:.2f})</td>
    <td>Top 7 ({result[6]['score']:.2f})</td>
    <td>Top 8 ({result[7]['score']:.2f})</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[4]['image_url']}"
            alt="score: {result[4]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[5]['image_url']}"
            alt="score: {result[5]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[6]['image_url']}"
            alt="score: {result[6]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[7]['image_url']}"
            alt="score: {result[7]['score']:.2f}">
    </td>
</tr>
</table>
            """
    return result_html
    
def retrieve(prompt):
    english_prompts = translate_ko2en([prompt])
    generated_images = generate_image(english_prompts)
    result_with_image = retrieve_image_with_image(generated_images)[0]
    result_with_text = retrieve_image_with_text([prompt])[0]
    # result_html = "\n".join([f"<img src='{ith['image_url']}' alt={ith['score']} />" for ith in result])
    
    result_with_image_html = get_html(result_with_image, caption="text-image-image") 
    result_with_text_html = get_html(result_with_text, caption="text-image")
    
    print("[*] result_with_image:", result_with_image)
    print("[*] result_with_text:", result_with_text)

    return english_prompts[0], generated_images[0], result_with_image_html, result_with_text_html

demo = gr.Interface(fn=retrieve, inputs="text", outputs= ["text", gr.Image(type="pil"), "html", "html"])
    
if __name__ == "__main__":
    demo.launch(show_api=False)   