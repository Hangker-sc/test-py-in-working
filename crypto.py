#读取文件内容
input_file_name='secret.daz'
with open(input_file_name, 'r', encoding='utf-8') as file:
        file_text= file.read()
        file_text=file_text.replace("X"," ").split()

#解密
text=""
text_conunt=0
id=2023090911013
for every_text in file_text:
    try:
        text_conunt+=1
        if len(every_text)==4:
            text+=bytes.fromhex(every_text).decode(encoding="utf-16be",errors="ignore")
        else:
            text += bytes.fromhex(every_text).decode(encoding="utf-8",errors="ignore")
    except:
        text_conunt-=1
        pass#非可见或者其他错误就不管了

#生成签名
signature=f'<解密人>{id}<情报总字数>{text_conunt}'
output_filename='interpretation.txt'
with open(output_filename,"w",encoding="utf-8") as f:
    f.write("\n")
    f.write(text)
    f.write("\n")
    f.write(signature)