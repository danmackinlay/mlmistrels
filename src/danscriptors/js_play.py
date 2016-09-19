
from IPython.display import Javascript

def init():
    return Javascript("""
require.config({
  paths: {
      tone: '//tonejs.github.io/CDN/latest/Tone'
  }
});
require(['tone'], (t)=>{
    window.Tone=t;
    element.append("<p>Tone Loaded</p>")
})
""")


def load(filepath):
    return Javascript("""
window.buffer=new Tone.Buffer(
    '{}',
        ()=>element.append(
            String(window.buffer.get().duration +
            " seconds loaded"
        )
    )
);""".format(filepath)
    )

def play():
    return Javascript("""
window.buffer=new Tone.Buffer(
    '{}',
        ()=>element.append(
            String(window.buffer.get().duration +
            " seconds loaded"
        )
    )
);""".format(filepath)
    )
