from config.extractors import kiwi_word_splitter


def tokens(text):
    return [t for t, _, _ in kiwi_word_splitter(text, lower=False)]


def test_value_tokens_match_in_context():
    for value, context in [("제6008공병단 정보과", "| 제6008공병단 정보과 | 원수호 병장 | 전산실 |"),
                           ("원수호 병장", "확인: 원수호 병장이 서명함"),
                           ("전산실", "전산실에서 반출함")]:
        inner, outer = tokens(value), tokens(context)
        assert any(outer[i:i + len(inner)] == inner for i in range(len(outer))), (value, inner, outer)


def test_offsets_point_into_text():
    text = "| 2026. 4. 7. | 장지후 병장이 |"
    for token, s, e in kiwi_word_splitter(text, lower=False):
        assert text[s:e] == token
