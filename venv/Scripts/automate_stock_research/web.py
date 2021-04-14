from flask import Flask

app = Flask(__name__)


@app.route('/')
@app.route('/hello')
@app.route('/hello/<user>')
def hello_world():
    return '''
    <!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title></title>
<script>
window.onload=function(){ with (new XMLHttpRequest()) {
  onreadystatechange=cb; open('GET','summaries.csv',true); responseType='text';send();
}}
function cb(){if(this.readyState===4)document.getElementById('main')
                                             .innerHTML=tbl(this.responseText); }
function tbl(csv){ // do whatever is necessary to create your table here ...
 return csv.split('\n')
           .map(function(tr,i){return '<tr><td>'
                                     +tr.replace(/\t/g,'</td><td>')
                                     +'</td></tr>';})
           .join('\n'); }
</script>
</head>
<body>
<h2>Hey, this is my fabulous "dynamic" html page!</h2>
<table id="main"></table>
</body>
</html>'''


if __name__ == '__main__':
    app.run()