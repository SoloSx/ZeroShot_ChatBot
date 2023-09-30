import datetime
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import MeCab
import ipadic

class TextClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R")
        self.model = AutoModelForSequenceClassification.from_pretrained("DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R")
        self.list_label = ["営業時間", "料金", "サービス内容", "予約方法", "社長"]
        self.day_label = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日", "今日","明日"]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.list_ABC = [x for x in string.ascii_uppercase]

        # main関数
    def check_text(self, text, list_label, shuffle=False):
        # question_flag を初期化
        question_flag = False
        # 形態素解析による単語チェック
        specific_words = ["今日","本日"]
        if self.question_Analysis(text, specific_words):
        #print("指定した単語が含まれています。")
            question_flag = True
        #=============================================
        #ラベルのリスト内の各要素の最後にピリオドがなければ追加。
        list_label = [x+'.' if x[-1] != '.' else x for x in list_label]
        #ラベルのリストをパディング（補完）
        list_label_new = list_label + [self.tokenizer.pad_token]* (20 - len(list_label))
        #print(list_label_new)
        if shuffle:
            random.shuffle(list_label_new)
        #ラベルとテキストを結合
        s_option = ' '.join(['('+self.list_ABC[i]+') '+list_label_new[i] for i in range(len(list_label_new))])
        text = f'{s_option} {self.tokenizer.sep_token} {text}'
        #モデルを評価モードに設定。
        self.model.to(self.device).eval()
        #テキストをトークン化
        encoding = self.tokenizer([text],truncation=True, max_length=512,return_tensors='pt')
        #cuda が有効ならGPU 無効ならCPUに送る
        item = {key: val.to(self.device) for key, val in encoding.items()}
        #モデルで予測
        logits = self.model(**item).logits
        logits = logits if shuffle else logits[:,0:len(list_label)]
        #ソフトマックス関数を適用して確率を計算。　->正規化され確率分布に変換される
        probs = torch.nn.functional.softmax(logits, dim = -1).tolist()
        #最も高い確率のラベルを選ぶ。
        predictions = torch.argmax(logits, dim=-1).item()
        probabilities = [round(x,5) for x in probs[0]]
        #print(probabilities)

        #加算処理
        if question_flag:
            search_words = ["今日.",]
            result = self.check_words_in_list(search_words,list_label_new)
        #print("加算処理結果：",result)
            if result:
                probabilities[7] += 0.5
                #print("更新後：",probabilities)
                logits[0][7] += 0.5  # 今日のラベルに対応する logits の値を更新
                # 予測の再計算
                predictions = torch.argmax(logits, dim=-1).item()

        print(f'prediction:    {predictions} => ({self.list_ABC[predictions]}) {list_label_new[predictions]}')
        print(f'probability:   {round(probabilities[predictions]*100,2)}%')
        return predictions  # predictions を返す


    def check_words_in_list(self, words, word_list):
        """
        複数の単語がリスト内に存在するかを確認する関数

        :param words: 確認したい単語のリスト
        :param word_list: 単語が含まれるリスト
        :return: 全ての単語がリスト内に存在する場合はTrue、それ以外はFalse
        """
        for word in words:
            if word not in word_list:
                return False
        return True

    # 使用例
    #my_list = ["apple", "banana", "cherry"]
    #day_label =  ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日","今日"]
    #search_words = ["banana", "apple"]
    #search_words = ["今日",]
    #result = check_words_in_list(search_words, my_list)
    #print(result)  # True
    #result = check_words_in_list(search_words, day_label)
    #print(result)  # True

    # 曜日取得用
    def date_weekday(self):
        weekday = datetime.date.today().weekday()
        #月曜
        if weekday == 0:
            return "の営業時間は10時から20時です。(月曜)"   
        #火曜
        elif weekday == 1:
            return "の営業時間は10時から20時です。(火曜)"
        #水曜
        elif weekday == 2:
            return "の営業時間は10時から20時です。(水曜)"
        #木曜
        elif weekday == 3:
            return "の営業時間は10時から20時です。(木曜)"
        #金曜
        elif weekday == 4:
            return "の営業時間は10時から20時です。(金曜)"
        #土曜
        elif weekday == 5:
            return "は休業日です。(土曜)"
        #日曜
        elif weekday == 6:
            return "は休業日です。(日曜)"
        else:
            return "不明です。"
    
    def check_content(self, prediction):
        time_flag = False
        if prediction == 0:
            time_flag = True
            #print("time_falag:",time_flag)
            return time_flag
        elif prediction == 1:
            print("料金についての情報はAです。")
        elif prediction == 2:
            print("サービス内容についてはBです。")
        elif prediction == 3:
            print("予約はCで行えます。")
        elif prediction == 4:
            print("社長ブログをご覧ください。（http:// ...）")
        else:
            print("どの条件にもマッチしませんでした。")

    def check_time(self, prediction):
        if prediction == 0:    #月曜日
            print("月曜日の営業時間は10時から20時です。")
        elif prediction == 1:  #火曜日
            print("火曜日の営業時間は10時から20時です。")
        elif prediction == 2:  #水曜日
            print("水曜日の営業時間は10時から20時です。")
        elif prediction == 3:  #木曜日
            print("木曜日の営業時間は10時から20時です。")
        elif prediction == 4:  #金曜日
            print("金曜日の営業時間は10時から20時です。")
        elif prediction == 5:  #土曜日
            print("土曜日は休業日です。")
        elif prediction == 6:  #日曜日
            print("日曜日は休業日です。")
        elif prediction == 7:  #今日
            print("今日" +  self.date_weekday())
        elif prediction == 8:  #明日
            #今日の曜日を整数として取得し、明日の曜日を返す
            today = datetime.datetime.now().weekday()
            tomorrow_weekday = (today + 1) % 7
            days = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
            if days[tomorrow_weekday] == "土曜" or days[tomorrow_weekday] == "日曜":
              print(f"明日は休業日です。({days[tomorrow_weekday]})")
            else:
              print(f"明日の営業時間は10時から20時です。({days[tomorrow_weekday]})")
            
        else:
            print("対象なし")
        prediction = None


    def question_Analysis(self, user_question, specific_words):
        #tagger = MeCab.Tagger()  # 「tagger = MeCab.Tagger('-d ' + unidic.DICDIR)」
        tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
        # MeCabを使用してテキストを解析
        result = tagger.parse(str(user_question))
        print(result)
        # 解析結果を分割し、単語ごとに処理
        for word_info in result.split("\n"):
            # 単語情報は「表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音」の形式
            word_info_parts = word_info.split("\t")

            if len(word_info_parts) > 1:
                # 単語部分を取得
                word = word_info_parts[0]

                # 特定の単語が含まれている場合
                for specific_word in specific_words:
                    if specific_word in word:
                        return True
        # 特定の単語が含まれていない場合
        return False
    
    def merge(self, text):
        # モデルを評価モードに設定
        self.model.to(self.device).eval()

        # ユーザーからの質問をプログラム実行時に入力
        user_question = text
        print("質問内容:", user_question)

        # 質問のラベルを予測
        list_predict = self.check_text(user_question, self.list_label)
        flag = self.check_content(list_predict)

        # 営業時間を表示
        if flag:
            day_predict = self.check_text(user_question, self.day_label)
            self.check_time(day_predict)
        print("---------------------------------------------------------------------------")

# 使用例
if __name__ == "__main__":
    text_classifier = TextClassifier()
    
    # ユーザーからの質問をコマンドラインから入力
    user_question = input("質問を入力してください: ")
    text_classifier.merge(user_question)