import base64
from io import BytesIO


from ko_to_en_translation import translate_ko2en
from text_to_image_generation import generate_image
from image_to_image_retrieval import retrieve_image_with_image

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str


ko_list = ["허리 작게 나온 일자 아이보리 면바지",\
           "살짝 A라인이고 무릎 아래까지 내려오는 캐쥬얼하게 입을 아이보리 면치마",\
           "진한 핑크색이면서 너무 여성스럽지도 않고 너무 캐쥬얼 하지도 않은 일자 미디 스커트",\
           "기본 일자핏 청바지",\
           "밑위길이가 길지만 반바지 길이는 짧은 밝은 색 청반바지",\
           "진한 핑크색이면서 너무 여성스럽지도 않고 너무 캐쥬얼 하지도 않은 일자 미디 스커트",\
           "여름에 시원하게 보일 화이트 반바지",\
            ]

english_list = translate_ko2en(ko_list)
generated_image_list = generate_image(english_list)
print("[*] generated_image_list[0]",  generated_image_list[0])
result_list = retrieve_image_with_image(generated_image_list)

base64_generated_image_list = [img_to_base64(img) for img in generated_image_list]
print(result_list)


# Save results
for i, (result, ko_des, en_des, base64_generated_image) in enumerate(zip(result_list, ko_list, english_list, base64_generated_image_list)):
    img_size=180
    HTML_STR = f"""
<table>

<tr>
    <td>KO Query</td>
    <td colspan="3">{ko_des}</td>
</tr>
<tr>
    <td>KO -> EN</td>
    <td colspan="3">{en_des}</td>
</tr>
<tr>
    <td>EN -> Generated Image</td>
    <td colspan="3"> <img height="{img_size}" width="{img_size}"
            src="data:image/JPEG;base64,{str(base64_generated_image)[2:-1]}"> </img></td>
</tr>
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
     

    with open(f"text_generated_image_to_image_retriever/output_markdown/result_{i}.html", "w") as f:
        f.write(HTML_STR)
    f.close()
