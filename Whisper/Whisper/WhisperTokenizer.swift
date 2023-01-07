//
//  WhisperTokenizer.swift
//  Whisper
//
//  Created by Anton Marini on 1/6/23.
//

//  Based heavily on https://github.com/huggingface/swift-coreml-transformers/blob/master/Sources/GPT2Tokenizer.swift
//  GPT2Tokenizer.swift by  Created by Julien Chaumond on 18/07/2019.

import Foundation
import CoreML
import Accelerate

struct Utils {
    /// Invert a (k, v) dictionary
    static func invert<K, V>(_ dict: Dictionary<K, V>) -> Dictionary<V, K> {
        var inverted: [V: K] = [:]
        for (k, v) in dict {
            inverted[v] = k
        }
        return inverted
    }
    
}

struct BytePair: Hashable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }
    init(tuple: [String]) {
        self.a = tuple[0]
        self.b = tuple[1]
    }
    
    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        return lhs.a == rhs.a && lhs.b == rhs.b
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

fileprivate extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
}


let byteEncoder: Dictionary<UTF8.CodeUnit, String> = [
    33: "!",
    34: "\"",
    35: "#",
    36: "$",
    37: "%",
    38: "&",
    39: "'",
    40: "(",
    41: ")",
    42: "*",
    43: "+",
    44: ",",
    45: "-",
    46: ".",
    47: "/",
    48: "0",
    49: "1",
    50: "2",
    51: "3",
    52: "4",
    53: "5",
    54: "6",
    55: "7",
    56: "8",
    57: "9",
    58: ":",
    59: ";",
    60: "<",
    61: "=",
    62: ">",
    63: "?",
    64: "@",
    65: "A",
    66: "B",
    67: "C",
    68: "D",
    69: "E",
    70: "F",
    71: "G",
    72: "H",
    73: "I",
    74: "J",
    75: "K",
    76: "L",
    77: "M",
    78: "N",
    79: "O",
    80: "P",
    81: "Q",
    82: "R",
    83: "S",
    84: "T",
    85: "U",
    86: "V",
    87: "W",
    88: "X",
    89: "Y",
    90: "Z",
    91: "[",
    92: "\\",
    93: "]",
    94: "^",
    95: "_",
    96: "`",
    97: "a",
    98: "b",
    99: "c",
    100: "d",
    101: "e",
    102: "f",
    103: "g",
    104: "h",
    105: "i",
    106: "j",
    107: "k",
    108: "l",
    109: "m",
    110: "n",
    111: "o",
    112: "p",
    113: "q",
    114: "r",
    115: "s",
    116: "t",
    117: "u",
    118: "v",
    119: "w",
    120: "x",
    121: "y",
    122: "z",
    123: "{",
    124: "|",
    125: "}",
    126: "~",
    161: "\u{00a1}",
    162: "\u{00a2}",
    163: "\u{00a3}",
    164: "\u{00a4}",
    165: "\u{00a5}",
    166: "\u{00a6}",
    167: "\u{00a7}",
    168: "\u{00a8}",
    169: "\u{00a9}",
    170: "\u{00aa}",
    171: "\u{00ab}",
    172: "\u{00ac}",
    174: "\u{00ae}",
    175: "\u{00af}",
    176: "\u{00b0}",
    177: "\u{00b1}",
    178: "\u{00b2}",
    179: "\u{00b3}",
    180: "\u{00b4}",
    181: "\u{00b5}",
    182: "\u{00b6}",
    183: "\u{00b7}",
    184: "\u{00b8}",
    185: "\u{00b9}",
    186: "\u{00ba}",
    187: "\u{00bb}",
    188: "\u{00bc}",
    189: "\u{00bd}",
    190: "\u{00be}",
    191: "\u{00bf}",
    192: "\u{00c0}",
    193: "\u{00c1}",
    194: "\u{00c2}",
    195: "\u{00c3}",
    196: "\u{00c4}",
    197: "\u{00c5}",
    198: "\u{00c6}",
    199: "\u{00c7}",
    200: "\u{00c8}",
    201: "\u{00c9}",
    202: "\u{00ca}",
    203: "\u{00cb}",
    204: "\u{00cc}",
    205: "\u{00cd}",
    206: "\u{00ce}",
    207: "\u{00cf}",
    208: "\u{00d0}",
    209: "\u{00d1}",
    210: "\u{00d2}",
    211: "\u{00d3}",
    212: "\u{00d4}",
    213: "\u{00d5}",
    214: "\u{00d6}",
    215: "\u{00d7}",
    216: "\u{00d8}",
    217: "\u{00d9}",
    218: "\u{00da}",
    219: "\u{00db}",
    220: "\u{00dc}",
    221: "\u{00dd}",
    222: "\u{00de}",
    223: "\u{00df}",
    224: "\u{00e0}",
    225: "\u{00e1}",
    226: "\u{00e2}",
    227: "\u{00e3}",
    228: "\u{00e4}",
    229: "\u{00e5}",
    230: "\u{00e6}",
    231: "\u{00e7}",
    232: "\u{00e8}",
    233: "\u{00e9}",
    234: "\u{00ea}",
    235: "\u{00eb}",
    236: "\u{00ec}",
    237: "\u{00ed}",
    238: "\u{00ee}",
    239: "\u{00ef}",
    240: "\u{00f0}",
    241: "\u{00f1}",
    242: "\u{00f2}",
    243: "\u{00f3}",
    244: "\u{00f4}",
    245: "\u{00f5}",
    246: "\u{00f6}",
    247: "\u{00f7}",
    248: "\u{00f8}",
    249: "\u{00f9}",
    250: "\u{00fa}",
    251: "\u{00fb}",
    252: "\u{00fc}",
    253: "\u{00fd}",
    254: "\u{00fe}",
    255: "\u{00ff}",
    0: "\u{0100}",
    1: "\u{0101}",
    2: "\u{0102}",
    3: "\u{0103}",
    4: "\u{0104}",
    5: "\u{0105}",
    6: "\u{0106}",
    7: "\u{0107}",
    8: "\u{0108}",
    9: "\u{0109}",
    10: "\u{010a}",
    11: "\u{010b}",
    12: "\u{010c}",
    13: "\u{010d}",
    14: "\u{010e}",
    15: "\u{010f}",
    16: "\u{0110}",
    17: "\u{0111}",
    18: "\u{0112}",
    19: "\u{0113}",
    20: "\u{0114}",
    21: "\u{0115}",
    22: "\u{0116}",
    23: "\u{0117}",
    24: "\u{0118}",
    25: "\u{0119}",
    26: "\u{011a}",
    27: "\u{011b}",
    28: "\u{011c}",
    29: "\u{011d}",
    30: "\u{011e}",
    31: "\u{011f}",
    32: "\u{0120}",
    127: "\u{0121}",
    128: "\u{0122}",
    129: "\u{0123}",
    130: "\u{0124}",
    131: "\u{0125}",
    132: "\u{0126}",
    133: "\u{0127}",
    134: "\u{0128}",
    135: "\u{0129}",
    136: "\u{012a}",
    137: "\u{012b}",
    138: "\u{012c}",
    139: "\u{012d}",
    140: "\u{012e}",
    141: "\u{012f}",
    142: "\u{0130}",
    143: "\u{0131}",
    144: "\u{0132}",
    145: "\u{0133}",
    146: "\u{0134}",
    147: "\u{0135}",
    148: "\u{0136}",
    149: "\u{0137}",
    150: "\u{0138}",
    151: "\u{0139}",
    152: "\u{013a}",
    153: "\u{013b}",
    154: "\u{013c}",
    155: "\u{013d}",
    156: "\u{013e}",
    157: "\u{013f}",
    158: "\u{0140}",
    159: "\u{0141}",
    160: "\u{0142}",
    173: "\u{0143}",
]

let byteDecoder = Utils.invert(byteEncoder)

class GPT2Tokenizer {
    let bpeRanks: Dictionary<BytePair, Int>
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    
    init() {
        let url = Bundle.main.url(forResource: "gpt2-merges", withExtension: "txt")!
        let bpeMergesTxt = try! String(contentsOf: url)
        let arr = bpeMergesTxt.split(separator: "\n").map { String($0) }
        var bpeRanks: Dictionary<BytePair, Int> = [:]
        for i in 1..<arr.count {
            let tuple = arr[i].split(separator: " ").map { String($0) }
            let bp = BytePair(tuple: tuple)
            bpeRanks[bp] = i - 1
        }
        self.bpeRanks = bpeRanks
        
        self.encoder = {
            let url = Bundle.main.url(forResource: "gpt2-vocab", withExtension: "json")!
            let json = try! Data(contentsOf: url)
            let decoder = JSONDecoder()
            let vocab = try! decoder.decode([String: Int].self, from: json)
            return vocab
        }()
        self.decoder = Utils.invert(self.encoder)
    }
    
    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { (token) -> String in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
    
    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count-1 {
            let bp = BytePair(
                word[i],
                word[i+1]
            )
            s.insert(bp)
        }
        return s
    }
    
    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }
        
        var word = Array(token).map { String($0) }
        var pairs = Array(getPairs(word: word))
        
        while true {
            let bigrams = pairs.filter { (bp) -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            let bigram = bigrams.min { (bp1, bp2) -> Bool in
                return bpeRanks[bp1]! < bpeRanks[bp2]!
            }!
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                
                if word[i] == first && i < word.count - 1 && word[i+1] == second {
                    newWord.append(first+second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in self.byteEncode(text: text) {
            let xx = self.bpe(token: token).split(separator: " ").map { String($0) }
            tokens.append(contentsOf: xx)
        }
        return tokens
    }
    
    /// Main entry point
    func encode(text: String) -> [Int] {
        return tokenize(text: text).map { encoder[$0]! }
    }
    
    /// Decode
    func decode(tokens: [Int]) -> String {
        let text = tokens.map { decoder[$0]! }.joined(separator: "")
        let utfCodepoints = text.map { byteDecoder[String($0)]! }
        return String(decoding: utfCodepoints, as: UTF8.self)
    }
}


class WhisperTokenizer:GPT2Tokenizer
{
    static let eotToken = 50256
    static let sotToken = 50257 // 50257 metalcpp or 50258?
    static let langToken = 50259 // sotToken + 1 + langIdx for a specific language, ie en is (langToken + 1)
    static let prevToken = 50360
    static let spolmToken = 50361 //?
    static let notToken = 50362
    static let begToken = 50363
    
    static let LANGUAGES = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]

    func tokenToMultiArray(token:Int) -> MLMultiArray
    {
        let array = try! MLMultiArray(shape: [1, 1], dataType: .float32)
        array[0] = NSNumber(integerLiteral: token)
        
        return array
    }

    func simdMaxIndexForRange(startToken:Int, endToken:Int, decoded:MLMultiArray) -> (Int, Float)
    {
        let confidence:[Float] = (startToken...endToken).map { decoded[$0].floatValue }

        var maxValue: Float = 0.0
        var maxIndex: vDSP_Length = 0

        vDSP_maxvi(confidence, 1, &maxValue, &maxIndex, vDSP_Length(confidence.count))
        
        return (Int(maxIndex), maxValue)

    }
    
   
    func predictLangToken(decoded:MLMultiArray) -> Int
    {
        let (token, _) = self.simdMaxIndexForRange(startToken: Self.langToken, endToken: 50357, decoded: decoded)
        return token

//        let confidence = (50259...50357).map { decoded[$0].floatValue }
//        let (langIdx, _) = confidence.enumerated().max { $0.element < $1.element }!
//
//        return langIdx
    }
    
    func langFromToken(token:Int) -> String
    {
        return Self.LANGUAGES[token]
    }
    
    func nextTokenGreedy(decoded:MLMultiArray) -> Int
    {
        let (token, _) = self.simdMaxIndexForRange(startToken: 0, endToken: Self.eotToken, decoded: decoded)
        return token

//        let confidence = (0...Self.eotToken).map {decoded[$0].floatValue }
//
//        let (tokenIdx, _) = confidence.enumerated().max { $0.element < $1.element }!
//        return tokenIdx
    }
}
