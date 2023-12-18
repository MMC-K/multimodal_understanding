from transformers import pipeline
from googletrans import Translator
translator = None


def translate_ko2en(text_list):
    global translator
    if translator is None:
        translator = pipeline('translation', model='NHNDQ/nllb-finetuned-ko2en', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
    output_list = [translator(text, max_length=512)[0]['translation_text'] for text in text_list]
    return output_list

google_translator = None
def google_translate_ko2en(text_list):
    global google_translator
    if google_translator is None:
        google_translator = Translator()
    output_list = [google_translator.translate(text, dest='en').text for text in text_list]
    return output_list

if __name__ == "__main__":
    # translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
    translator = pipeline('translation', model='NHNDQ/nllb-finetuned-ko2en', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
    text = '에스닉 무늬의 짧은 기장 원피스'
    output = translator(text, max_length=512)
    print(output[0]['translation_text'])

    print(google_translate_ko2en([text]))

