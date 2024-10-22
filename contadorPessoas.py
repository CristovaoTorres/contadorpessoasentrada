import cv2
import uuid

# Carregar o vídeo
video = cv2.VideoCapture('loja.mp4')

# Criar o detector de pessoas usando HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Inicializar variáveis
contador = 0
pessoas_rastreio = {}
x, y, w, h = 340, 250, 150, 300  # Coordenadas da área da porta
max_distance = 50  # Distância máxima para considerar que é a mesma pessoa
tempo_maximo = 20  # Tempo máximo que a pessoa pode permanecer na área (em frames)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1100, 720))

    # Detectar pessoas no frame inteiro
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Atualizar lista de rastreamento para novas pessoas
    novas_pessoas_rastreio = {}
    for (px, py, pw, ph) in boxes:
        # Calcular o centro do bounding box da pessoa detectada
        centro_x = px + pw // 2
        centro_y = py + ph // 2
        pessoa_contada = False

        # Verificar se o centro da pessoa está dentro da região da porta
        if (x < centro_x < x + w) and (y < centro_y < y + h):
            # Procurar pessoas na lista de rastreamento
            for id_pessoa, (rastreio_x, rastreio_y, frames) in pessoas_rastreio.items():
                distancia = ((rastreio_x - centro_x) ** 2 + (rastreio_y - centro_y) ** 2) ** 0.5
                
                # Se a pessoa está próxima o suficiente e ainda está na região, atualizar o rastreamento
                if distancia < max_distance:
                    novas_pessoas_rastreio[id_pessoa] = (centro_x, centro_y, frames + 1)
                    pessoa_contada = True
                    break

            # Se a pessoa não estava no rastreamento, é uma nova pessoa entrando
            if not pessoa_contada:
                # Gerar um ID único para a nova pessoa
                id_novo = str(uuid.uuid4())
                novas_pessoas_rastreio[id_novo] = (centro_x, centro_y, 0)
                contador += 1

            # Desenhar a caixa delimitadora ao redor das pessoas detectadas
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

    # Filtrar pessoas que já saíram da área ou excederam o tempo máximo
    pessoas_rastreio = {id_pessoa: dados for id_pessoa, dados in novas_pessoas_rastreio.items() if dados[2] < tempo_maximo}

    # Desenhar a região da porta na imagem
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mostrar o contador de pessoas
    cv2.putText(frame, f"Contador: {contador}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibir os frames
    cv2.imshow('Detecção de Pessoas', frame)

    # Sair quando 'q' for pressionado
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()
