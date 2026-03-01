# Assistente de Voz com Whisper, Gemini e gTTS no Google Colab

Este notebook implementa um assistente de voz simples utilizando as seguintes tecnologias:

*   **Whisper (OpenAI)**: Para Transcrição de Fala (Speech-to-Text).
*   **Google Gemini API**: Para Geração de Respostas (Text Generation).
*   **gTTS (Google Text-to-Speech)**: Para Síntese de Fala (Text-to-Speech).

## Como Funciona

1.  **Gravação de Áudio**: O notebook grava alguns segundos de áudio do microfone do usuário no navegador (usando JavaScript).
2.  **Transcrição com Whisper**: O áudio gravado é transcrito para texto usando o modelo `small` do Whisper.
3.  **Geração de Resposta com Gemini**: A transcrição é enviada para a API do Google Gemini, que gera uma resposta textual.
4.  **Síntese de Voz com gTTS**: A resposta do Gemini é convertida em áudio usando gTTS e reproduzida para o usuário.

## Estrutura do Código

### 1. Gravação de Áudio

```python
# Referência: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be

from IPython.display import Audio, display, Javascript
from google.colab import output
from base64 import b64decode

# Código JavaScript para gravar áudio do usuário usando a "MediaStream Recording API"
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec=5):
  # Executa o código JavaScript para gravar o áudio
  display(Javascript(RECORD))
  # Recebe o áudio gravado como resultado do JavaScript
  js_result = output.eval_js('record(%s)' % (sec * 1000))
   # Decodifica o áudio em base64
  audio = b64decode(js_result.split(',')[1])
  # Salva o áudio em um arquivo
  file_name = 'request_audio.wav'
  with open(file_name, 'wb') as f:
    f.write(audio)
  # Retorna o caminho do arquivo de áudio (pasta padrão do Google Colab)
  return f'/content/{file_name}'

# Grava o áudio do usuário por um tempo determinado (padrão 5 segundos)
print('Ouvindo...\n')
record_file = record()

# Exibe o áudio gravado
display(Audio(record_file, autoplay=False))
```

Esta seção define uma função `record()` que usa JavaScript para capturar áudio do microfone do usuário diretamente no navegador. O áudio é codificado em Base64, enviado de volta ao Python, decodificado e salvo como um arquivo `.wav`.

### 2. Reconhecimento de Fala com Whisper

```python
!pip install git+https://github.com/openai/whisper.git -q
import whisper

# Selecione o modelo do Whisper que melhor atenda às suas necessidades:
# https://github.com/openai/whisper#available-models-and-languages
model = whisper.load_model("small")

# Transcreve o audio gravado anteriormente.
# A variável 'language' é definida no início do notebook (ex: 'pt' para português).
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]
print(transcription)
```

Instala a biblioteca Whisper e carrega um modelo de transcrição (`small`). Em seguida, a função `model.transcribe()` é usada para converter o arquivo de áudio `record_file` em texto, armazenando o resultado na variável `transcription`.

### 3. Integração com a API do Gemini (Google)

Esta seção lida com a configuração e chamada da API do Google Gemini para gerar respostas. É crucial configurar a chave de API de forma segura.

```python
# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
from google.colab import userdata

# Carrega a chave de API do Google Gemini a partir dos segredos do Colab
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Inicializa o modelo Gemini
gemini_model = genai.GenerativeModel('gemini-pro-latest') # Ou outro modelo disponível como 'gemini-pro'

# Envia a transcrição do áudio para o modelo Gemini
try:
    gemini_response = gemini_model.generate_content(transcription)
    # Obtém a resposta gerada pelo Gemini
    chatgpt_response = gemini_response.text
    print(chatgpt_response)
except Exception as e:
    chatgpt_response = f"Erro ao gerar resposta do Gemini: {e}"
    print(chatgpt_response)
```

#### Como Adicionar a Chave de API do Gemini nos Segredos do Colab

Para usar a API do Google Gemini, você precisa de uma chave de API. **É altamente recomendável não expor sua chave de API diretamente no código.** O Google Colab oferece um gerenciador de segredos para isso:

1.  **Obtenha sua Chave de API**: Se você não tem uma, crie uma em [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  **Abra o Gerenciador de Segredos no Colab**: No painel esquerdo do seu notebook Colab, clique no ícone de uma chave (🔑). Este é o gerenciador de Segredos.
3.  **Adicione um Novo Segredo**: Clique em `+ New secret`.
4.  **Defina o Nome e o Valor**: No campo 'Name', digite `GOOGLE_API_KEY` (este é o nome que o código usa para buscar a chave). No campo 'Value', cole sua chave de API do Gemini. Certifique-se de ativar a opção `Notebook access` para que o notebook possa acessá-la.

Com isso, `userdata.get('GOOGLE_API_KEY')` buscará sua chave de forma segura.

### 4. Síntese de Resposta com gTTS

```python
!pip install gTTS
from gtts import gTTS

# Cria um objeto gTTS com a resposta gerada pelo Gemini e a língua que será sintetizada em voz.
gtts_object = gTTS(text=chatgpt_response, lang=language, slow=False)

# Salva o áudio da resposta no arquivo especificado
response_audio = "/content/response_audio.wav"
gtts_object.save(response_audio)

# Reproduz o áudio da resposta salvo no arquivo
display(Audio(response_audio, autoplay=True))
```

Esta seção instala a biblioteca gTTS, usa a `chatgpt_response` (que pode ser a resposta do Gemini ou uma mensagem de erro) e a variável `language` (definida como 'pt' no início do notebook) para sintetizar o texto em um arquivo de áudio MP3, que é salvo e depois reproduzido.
