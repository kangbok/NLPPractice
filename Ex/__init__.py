from konlpy.tag import Kkma, Twitter
from konlpy.utils import pprint

kkma = Kkma()
twitter = Twitter()

# pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
# pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
# pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))
# pprint(kkma.pos(u'나는 신세계의 왕이 될 자이다. 그는 나에게로 와 꽃이 되었다.'))
poslist = kkma.pos(u'나는 컴퓨터 개발자 입니다. 그는 나에게로 와 꽃이 되었다.')
tlist = twitter.pos("나는 컴퓨터 개발자 입니다. 그는 나에게로 와 꽃이 되었다.")

# for s in poslist:
#     # s = s.split(",")
#     print(s)
#
# print("=" * 20);
#
# for s in tlist:
#     # s = s.split(",")
#     print(s)

print(twitter.morphs("나는 컴퓨터 개발자 입니다. 그는 나에게로 와 꽃이 되었다."))