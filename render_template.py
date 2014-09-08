import jinja2
import sys

def m_gen(filename="bomze.txt"):
    handle = open(filename)
    for line in handle:
        a,b,c,d,e,f,g,h,i = map(int, line.split())
        yield """\\left(\\begin{matrix}
        %s & %s & %s \\\\
        %s & %s & %s \\\\
        %s & %s & %s
        \\end{matrix} \\right)""" % (a,b,c,d,e,f,g,h,i)

if __name__ == "__main__":
    template_filename = "template.html"
    out_filename = "plots.html"
    try:
        template_filename = sys.argv[1]
        out_filename = sys.argv[2]
    except IndexError:
        pass

template_loader = jinja2.FileSystemLoader("./")
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template(template_filename)

#print list(enumerate(m_gen()))[0]

output = template.render(ms=list(enumerate(m_gen())))
f = open(out_filename,'w')
f.write(output)