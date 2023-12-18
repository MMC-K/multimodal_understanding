import sys
import os
import json
import glob
import base64
from PIL import Image
from io import BytesIO
import numpy as np

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def convert_to_html(json_object, image=None, num_show=5):

    HTML_TEXT = f"<table> <tr> <td>Korean</td> <td colspan='{num_show}'>{json_object['korean']}</td> </tr>"
    
    # json file with text to image retrieval
    if image is not None:
        HTML_TEXT += f"<tr><td>English</td> <td colspan='{num_show}'>{json_object['english']}</td></tr>"
        HTML_TEXT += f"<tr><td>Generated image</td> <td colspan='{num_show}'><img width=180 height=180 src=data:image/JPEG;base64,{str(img_to_base64(image))[2:-1]} /> </td></tr>"

    HTML_TEXT += f"<tr><td>requested_tags</td> <td colspan='{num_show}'>{json_object['request_tags']}</td></tr>"
    def create_tr(key, pre_value="", post_value=""):
        html = f"<tr> <td>{key}</td>"
        for i in range(len(json_object["retrieved_urls"])):
            if i >= num_show:
                break
            html += f"<td> {pre_value} {json_object[key][i]} {post_value}  </td>"
        html += "</tr>"
        return html

    
    HTML_TEXT += create_tr("retrieved_tags_list")
    HTML_TEXT += create_tr("intersection_tags_list")
    HTML_TEXT += create_tr("recall_list")
    HTML_TEXT += create_tr("precision_list")
    HTML_TEXT += create_tr("retrieved_urls", "<img src=", "width=180 height=180 />")
    HTML_TEXT += f"<tr><td>Reference image</td> <td colspan='{num_show}'><img width=180 height=180 src={json_object['reference_url']} /> </td></tr>"
    HTML_TEXT += "</table>"

    return HTML_TEXT

result_path = sys.argv[1]


for json_path in glob.glob(os.path.join(result_path, "*.json")):
    json_object = json.load(open(json_path, mode="r"))
    #compute intersections
    intersection_tags_list = []
    for retrieved_tags in json_object["retrieved_tags_list"]:
        intersection_tags_list.append(list(set(json_object["request_tags"]).intersection(set(retrieved_tags))))
    json_object["intersection_tags_list"] = intersection_tags_list

    image = None
    img_path = os.path.join(result_path, os.path.basename(json_path).replace(".json", ".jpg") )
    if os.path.exists(img_path):
        image = Image.open(img_path)
    HTML = convert_to_html(json_object, image=image)
    
    mean_recall = np.mean(json_object['recall_list'])

    save_path = os.path.join(result_path, os.path.basename(json_path).replace(".json", "_{:.02f}.md".format(mean_recall)) )
    HTML_file = open(save_path, mode="w")
    HTML_file.write(HTML)
    HTML_file.close()
    print(save_path)


