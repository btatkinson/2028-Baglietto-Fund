import re
t = open("out/scorers.html",encoding="utf-8").read()
# extract first game header + first team panel header + total row
m = re.search(r"<section>.*?</section>", t, re.S).group(0)
# strip tags lightly for readability
import html as H
def show(frag):
    frag = re.sub(r"<(h2|div class='th'|table|tr|section)[^>]*>", "\n", frag)
    frag = re.sub(r"<[^>]+>"," | ",frag)
    frag = re.sub(r"\|\s*\|"," | ",frag)
    return H.unescape(re.sub(r"[ \t]*\|[ \t]*\|+"," | ",frag))
print(show(m)[:1400])
