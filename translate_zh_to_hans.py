from langconv import *

def Traditional2Simplified(sentence):
    
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def Simplified2Traditional(sentence):
    
    sentence = Converter('zh-hant').convert(sentence)
    return sentence

if __name__=="__main__":
    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = Traditional2Simplified(traditional_sentence)
    print(simplified_sentence)
