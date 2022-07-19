# Import các thư viện
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from underthesea import word_tokenize
from keras.layers import Embedding, Dense, LSTM
from keras.models import Model
from keras.layers import Input
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import pickle
from keras.models import load_model
from keras.models import Model
warnings.filterwarnings('ignore')


# Tạo mảng để lưu danh sách câu hỏi và câu trả lời từ tập dataset
questions_chatbot = []
answers_chatbot = []

# Mảng lưu đường dẫn tương đối đến các tập dữ liệu Chatbot
all_dataset = ['./dataset/bạn bè.txt', './dataset/các câu hỏi phức tạp.txt', './dataset/đất nước.txt', './dataset/địa chỉ.txt',
'./dataset/du lịch.txt', './dataset/gia đình.txt', './dataset/giải trí.txt', './dataset/học tập.txt', './dataset/nghề nghiệp.txt',
'./dataset/nghỉ lễ.txt', './dataset/người yêu.txt', './dataset/robot.txt', './dataset/shoping.txt', './dataset/tán gẫu.txt',
'./dataset/tdtu.txt', './dataset/thông tin cá nhân.txt', './dataset/trò chuyện về đi ăn.txt']

# Duyệt qua tất cả các tệp trong mảng all_dataset
total_dataset = len(all_dataset) # Đếm tất cả các đường dẫn trong mảng all_dataset
for i in range(total_dataset): # Thưc hiện vòng lặp tương ứng với số lượng đường dẫn mà ta đã đếm ở trên
    with open(all_dataset[i], encoding='UTF-8') as txt: # Tiến hành mở tệp dữ liệU Chatbot tương ứng với số thứ tự hiện tại đang duyệ của vòng lặp
        lines = txt.readlines() # Lưu tất cả các dòng vào trong mảng lines
        for line in lines: # Duyệt qua từng dòng trong mảng lines
            tmp = line.split("__eou__") # Tạo mảng tmp lưu hai giá trị được cắt tại vị trí có chuỗi "__eou__"
            question = tmp[0].strip() # Gán giá trị đầu tiên trong mảng tmp tương ứng với câu hỏi và xóa các khoảng trắng
            answer = tmp[1].strip() # Gán giá trị thứ hai trong mảng tmp tương ứng với câu trả lời và xóa các khoảng trắng
            questions_chatbot.append(question) # Thêm câu hỏi được lấy ra vào mảng questions_chatbot
            answers_chatbot.append(answer) # Thêm câu trả lời được lấy ra vào mảng answers_chatbot

# Hàm chuẩn hóa được sử dụng chuẩn hóa các câu trong tập dữ liệu và chuẩn hóa tin nhắn mà người dùng nhập vào
def normalize_text(sentence):
    punctuation_dict = {"``","''", '?', '*', ']',':', '(', ')', '“', '”','"', '!', '&', ';', '>', '…', '’', '.', ',', '...', '-',} # Định nghĩa một tập danh sách các dấu câu
    
    # Chuyển về chữ thường để đồng nhất tất cả
    sentence = sentence.lower()
    
    # Lược bỏ các dấu câu
    sentence = [char for char in sentence if char not in punctuation_dict] # Duyệt qua từng từ trong câu nếu ký tự không nằm trong danh sách các ký tự được định nghĩa thì lưu vào mảng sentence
    sentence = ''.join(sentence) # Thực hiện nối các từ trong mảng sentence thành một câu
    
    # Thay thế các khoảng trắng
    sentence = sentence.replace("   ", " ") # Thay thế 3 khoảng trắng thành 1 khoảng trắng
    sentence = sentence.replace("  ", " ") # Thay thế 2 khoảng trắng thành 1 khoảng trắng
    sentence = sentence.strip() # Xóa các khoảng trắng đầu và cuối của câu
    
    # Tách từ trong từng câu bằng thư viện Underthesea
    sentence = word_tokenize(sentence)
    
    return sentence # Trả về câu đã được chuẩn hóa

# Xóa question và answer khi answer empty string
list_index_of_empty_answer = [] # Tạo một mảng để lưu các giá trị tương ứng với vị trí của câu hỏi không có câu trả lời
total_answer = len(answers_chatbot) # Đếm tất cả các câu trả lời trong tập dữ liệu
for i in range(total_answer): # Thực hiện vòng lặp tương ứng với số câu trả lời
    if(answers_chatbot[i] == ""): # Nếu câu trả lời là rỗng
        list_index_of_empty_answer.append(i) # Thêm vị trí của câu trả lời đó trong mảng answers_chatbot

# Định nghĩa hàm xóa các câu trả lời trống với đầu vào là mảng lưu vị trí của các câu trả lời trống
def delete_empty_answer(list_index):
    total_index = len(list_index) # Đếm tổng số câu trả lời trống
    for i in range(total_index): # Thực hiện vòng lặp tương ứng với số câu trả lời trống
        del questions_chatbot[list_index[i] - i] # Thực hiện xóa câu hỏi tại vị trí thứ i
        del answers_chatbot[list_index[i] - i] # Thực hiện xóa câu trả lời tại vị trí thứ i
delete_empty_answer(list_index_of_empty_answer) # Thực hiện xóa

total_question = len(questions_chatbot) # Đếm tổng số câu hỏi trong mảng questions_chatbot
# Thực hiện vòng lặp với số lần tương ứng với tổng số câu hỏi
for i in range(total_question):
    questions_chatbot[i] = normalize_text(questions_chatbot[i]) # Thực hiện cập nhật câu hỏi tại vị trí thứ i thành câu đã được chuẩn hóa qua hàm normalize_text
    answers_chatbot[i] = normalize_text(answers_chatbot[i]) # Thực hiện cập nhật câu hỏi tại vị trí thứ i thành câu đã được chuẩn hóa qua hàm normalize_text
    
data = questions_chatbot # Gán mảng câu hỏi questions_chatbot cho biến data
data_answer = answers_chatbot # Gán mảng câu trả lời answers_chatbot cho biến data_answer

# Tạo biến word2count dạng Dict để lưu trữ số lần xuất hiện của một từ
word2count = {}

# Thực hiện vòng lặp để khởi tạo giá trị cho biến word2count
for line in data: # Thực hiện vòng lặp duyệt qua mảng câu hỏi
    for word in line:  # Duyệt qua từng từ trong câu
        if word not in word2count: # Nếu từ chưa có trong dict word2count
            word2count[word] = 1 # Tiến hành thêm giá trị Key tương ứng với từ đó và khởi tạo giá trị biến đếm số lần xuất hiện value của nó là 1
        else:
            word2count[word] += 1 # Nếu từ đó đã xuất hiện thì tiến hành cộng thêm 1 mỗi lần có từ trùng được duyệt qua

# Thực hiện tương tự cho mảng câu trả lời
for line in data_answer:
    for word in line:
        if word not in word2count: # Nếu từ chưa có trong từ điển
            word2count[word] = 1 # Khởi tạo giá trị 1 nếu chưa từ chưa có trong từ điển
        else:
            word2count[word] += 1 # Cộng thêm 1 nếu chưa từ chưa có trong từ điển

temp1 = [] # Tạo mảng temp để lưu trữ giá trị câu trả được gắn tag

for i in data_answer: # Duyệt qua từng câu trong mảng giá trị
  i.insert(0, "<SOS>") # Chèn tag <SOS> vào đầu câu
  i.insert(len(i),"<EOS>")  # Chèn tag <EOS> vào cuối câu
  temp1.append(i) # Thêm câu được gắn tag vào mảng temp1

# Tạo biến để chuyển đổi từ thành các số
word2index = {}
# Khởi tạo biến đếm
word_number = 0

# Duyệt qua từng từ và gán thứ tự cho mỗi từ
for word, _ in word2count.items():
    word2index[word] = word_number # Khởi tạo gán giá trị cho mỗi từ đươc duyệt qua
    word_number += 1 # Tăng biến đếm lên 1 đơn vị

# Thêm 3 tag <EOS>, <SOS>, <OUT> vào từ điển
tokens = ['<EOS>', '<SOS>', '<OUT>'] # Khởi tạo mảng tokens lưu 3 giá trị tag

VOCAB_SIZE = len(word2index) # Đếm tổng số từ trong biến word2index

for token in tokens: # Duyệt qua từng tag trong mảng token
    word2index[token] = VOCAB_SIZE # Thêm vào biến dict word2index giá trị 3 tag tương ứng với giá trị VOCAB_SIZE
    VOCAB_SIZE += 1 # Tăng giá trị VOCAB_SIZE lên 1

# Khởi tạo biến index2word để các giá trị số đại diện cho các từ
index2word = {w:v for v, w in word2index.items()} # Thực hiện đảo vị trí giá trị key và value và gán cho biến index2word

# Chuyển các câu trong question thành vector số
encoder_input_data = [] # Tạo mảng lưu các giá trị 

for line in data: # Duyệt qua từng câu hỏi trong mảng data
    tmp = [] # Biến tạm để lưu giá trị
    for word in line: # Duyệt qua từng từ trong câu
        if word not in word2index: # Nếu từ không có trong biến word2index
            tmp.append(word2index['<OUT>']) # Thêm giá trị tag <OUT> vào trong mảng tmp
        else:
            tmp.append(word2index[word]) # Nếu có xuất hiện thì thêm giá trị value tương ứng với từ đó
        
    encoder_input_data.append(tmp) # Thêm vào mảng encoder_input_data giá trị của mảng tmp tương ứng với câu được duyệt

decoder_input_data = [] # Tạo mảng để lưu các giá trị

for line in temp1: # Duyệt qua các câu trả lời đã được gắn tag
    tmp = [] # tạo mảng tmp lưu các giá trị của câu
    for word in line: # Duyệt qua từng từ trong câu
        if word not in word2index: # Nếu từ không có trong từ điển word2index
            tmp.append(word2index['<OUT>']) # Thêm tag <OUT> vào tại vị trí đó
        else:
            tmp.append(word2index[word]) # Thêm giá trị số tương ứng với từ đó trong biến word2index
            
    decoder_input_data.append(tmp) # Thực hiện thêm các câu vào mảng decoder_input_data

# Xác định chiều dài tối đa của câu
MAX_LEN = 20

# Padding độ dài của câu hỏi và trả lời đã được vector hóa sao cho độ dài bằng nhau
encoder_input_data = pad_sequences(encoder_input_data, MAX_LEN, padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_input_data, MAX_LEN, padding='post', truncating='post')

# Kết quả đầu ra
decoder_target_data = [] # Tạo mảng lưu kết quả đầu ra
for decoder_input in decoder_input_data: # Duyệt qua các câu trả lời đã được chuyển thành vector số
    decoder_target_data.append(decoder_input[1:]) # Loại bỏ Tag đầu tiên là <SOS>
    
# Padding vector đầu ra sao cho bằng MAX_LEN
decoder_target_data = pad_sequences(decoder_target_data, MAX_LEN, padding='post', truncating='post') # Padding lại sau khi loại bỏ tag <SOS>

# Trích xuất ra các đặc trưng (Sample, Timestep, Feature) cho LSTM Model
decoder_target_data = to_categorical(decoder_target_data, len(word2index))

# Khai báo các giá trị đầu vào cho các Layer
VOCAB_SIZE = len(word2index)
HIDDEN_DIM = 300
INPUT_DIM = VOCAB_SIZE + 1
embedding_dimention = 100

# Định nghĩa lớp Embedding
embed = Embedding(input_dim = INPUT_DIM, output_dim=embedding_dimention, input_length=MAX_LEN, trainable=True)

# Định nghĩa lớp Encoder của LSTM
# Dùng để xử lý chuỗi đầu vào và trả về state của nó
encoder_inputs = Input(shape=(MAX_LEN, ))
encoder_embed = embed(encoder_inputs)
encoder_lstm = LSTM(HIDDEN_DIM, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)

# Giữ lại giá trị State_h và State_c đóng vai trò là đầu vào cho bước tiếp theo
encoder_states = [state_h, state_c]

# Định nghĩa lớp Decoder
# Được train để predict các từ tiếp theo cho các từ trước đó của decoder_target_data
decoder_inputs = Input(shape=(MAX_LEN, ))
decoder_embed = embed(decoder_inputs)
decoder_lstm = LSTM(HIDDEN_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)

# Định nghĩa lớp Dense đầu ra ứng với tổng số từ trong từ điển
dense = Dense(VOCAB_SIZE, activation='softmax')

# Lấy ra giá trị từ lớp Decoder LSTM
output = dense(decoder_outputs)

# Định nghĩa Model chung
model1 = Model([encoder_inputs, decoder_inputs], output)

# Compile Model
model1.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

# Lưu biến từ điển word2index
with open('word2index.pkl', 'wb') as f:
    pickle.dump(word2index, f)

# Lưu biến index2word
with open('index2word.pkl', 'wb') as f:
    pickle.dump(index2word, f)

# Load Model đã Train để tiết kiệm thời gian Tranning
model1 = load_model('./LSTM_ChatBot_Final.h5')

# Xem tóm tắt lại mô hình
model1.summary()

# Load lại hai biến Dict là word2index và index2word để phục vụ cho việc chuyển đổi của mô hình Seq2Seq
with open('word2index.pkl', 'rb') as f:
    word2index = pickle.load(f)

with open('index2word.pkl', 'rb') as f:
    index2word = pickle.load(f)


# Khai báo số chiều và độ dài tối đa cho mỗi message
HIDDEN_DIM = 300
MAX_LEN = 20


# Load lớp Input cho Encoder
input_encoder = model1.input[0] 

# Load lớp Input cho bộ Decoder
input_decoder = model1.input[1]


# Load lớp Embbeing
embed = model1.layers[2]

# Load lớp LSTM cho Encoder
output_encoder, state_h, state_c = model1.layers[3].output 

# Load lớp LSTM cho Decoder
decoder_lstm = model1.layers[4]

#Load lớp Dense
dense = model1.layers[5]

#Load Encoder Model
# Mảng lưu hai trạng thái state_h và state_c của Encoder
encoder_states = [state_h, state_c]

# Model cho đầu vào của Message và trả về các final state vector của bộ Encoder LSTM
encoder_model = Model([input_encoder], encoder_states)

# Decoder Model dùng để dự đoán suy luận từ
decoder_state_input_h = Input(shape=(HIDDEN_DIM,))
decoder_state_input_c = Input(shape=(HIDDEN_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Lấy ra các state từ Encoder
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(embed(input_decoder), initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]

# Định nghĩa Decoder Model
decoder_model = Model([input_decoder] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Hàm thực hiện quá trình xử lý tin nhắn đầu vào
def chat(message):
    
    question_input = message # Gán tin nhắn đầu vào cho biến question_input
    
    if question_input.lower() == "quit" or question_input.lower() == "stop": # Check điều kiện để đưa ra lời chào tạm biệt
        end_question = "Xin chào bạn hẹn bạn gặp lại vào lần sau" # Khởi tạo biến lưu câu chào tạm biệt
        return end_question # Trả về câu chào
    else:
        question_input = normalize_text(question_input) # Thực hiện chuẩn hóa tin nhắn từ phía người dùng về đúng dạng mà mô hình NLP đã được xây dựng có thể xử lý được
        question_input_list = [question_input] # Thêm tin nhắn vào mảng để thực hiện cho các vòng lặp
        question_vector = [] # Tạo mảng để chưa các từ trong câu hỏi
        
        # Chuyển đổi message đầu vào thành dạng vector
        for x in question_input_list: # Duyệt qua các câu trong mảng câu hỏi đầu vào
            temp = [] # Tạo mảng để lưu trữ các giá trị được chuyển đổi về dạng vector
            for y in x: # Duyệt qua từng từ
                try:
                    temp.append(word2index[y]) # Thêm vào mảng temp các giá trị value tương ứng với key đầu vào là từ của message trong biến Dict word2index
                except:
                    temp.append(word2index['<OUT>']) # Nếu từ không có trong từ điển thì vào temp value của tag <OUT> trong Dict word2index
            question_vector.append(temp) # Thêm vào mảng question_vector
        
        # Padding độ dài vector của tin nhắn đầu vào sao cho đồng nhất với mô hình (MAX_LEN = 20)
        question_vector = pad_sequences(question_vector, MAX_LEN, padding='post')
        
        # Truyền câu hỏi vào cho bộ Encoder LSTM để trả về các state của bộ Encoder LSTM
        stat = encoder_model.predict(question_vector)
        
        # Tạo một chuỗi mục tiêu có độ dài 1 và gán cho nó là giá trị index của <SOS> trong Dict word2index
        empty_target_seq = np.zeros((1,1)) # 
        empty_target_seq[0, 0] = word2index['<SOS>']
        
        # Điều kiện dừng cho bộ Decoder
        stop_condition = False
        # Tạo môt biến để lưu kết quả cuối cùng
        decoded_translation = ''

        while not stop_condition:
            # Lấy ra prediction từ state của encoder và index trước đó
            dec_outputs , h, c= decoder_model.predict([empty_target_seq] + stat)
            decoder_concat_input = dense(dec_outputs)
            
            # Từ giá trị prediction ở trên, ta lấy ra giá trị xác suất lớn nhất
            sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
            
            # Thực hiện nối từ tương ứng với giá trị index trong biến Dict index2word vào biến sampled_word
            sampled_word = index2word[sampled_word_index] + ' '
            
            # Nếu word không phải EOS thì nối vào answer
            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word
            
            # Nếu giá trị của biến word là EOS hoặc độ dài của vượt quá 20 thì dừng việc dự đoán
            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > MAX_LEN+1:
                stop_condition = True
            
            # Khởi tạo lại empty_target_seq
            # Và tiến hành gán giá trị hiện tại của từ đươc dự đoán cho empty_target_seq
            empty_target_seq = np.zeros((1,1))
            empty_target_seq[0,0] = sampled_word_index
            
            # Cập nhật State
            stat = [h, c]
        return decoded_translation

# Django Rest Framework
# Định nghĩa lớp API để nhận dữ liệu được gửi lên từ phía Client
class APIChatVietNamese(APIView):
    def post(self, request, pk=None): # Định nghĩa phương thức POST
        if pk == " " or pk == None: # Nếu tin nhắn đầu vào là dấu khoảng trắng hoặc rỗng thì trả về thông điệp yêu cầu nhập tin nhắn.
            return Response({'isvalid':True, 'id':'Bạn vui lòng nhập tin nhắn và không được bỏ trống nhé !'}, status=status.HTTP_200_OK)
        result = chat(pk).replace("_", " ") # Nếu không thì thực hiện cho tin nhắn đi qua hàm chat để truy xuất mô hình Seq2Seq
        if result == "": # Nếu kết quả trả về là rỗng thì trả về thông báo
            return Response({'isvalid':True, 'id':'Mình chưa hiểu rõ. Bạn có thể hỏi lại được không?'}, status=status.HTTP_200_OK)
        return Response({'isvalid':True, 'id':result}, status=status.HTTP_200_OK) # Nếu hợp lệ thì trả về kết quả là biến result.

    def get(self, request, pk=None): # Định nghĩa phương thức GET
        if not pk:
            return Response({'isvalid':False, 'msg':'Tin nhắn không hợp lệ'}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'isvalid':True, 'status':pk}, status=status.HTTP_200_OK)


