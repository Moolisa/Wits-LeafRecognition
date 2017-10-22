var socket = io('http://localhost:3500');
//var socket = io('http://10.0.0.3:3500'); //Fix this so that it auto-connects
var uploader = new SocketIOFileClient(socket);

var form = document.getElementById('form');
var resDisp = document.getElementById('myPicture');
var picDisp = document.getElementById('pic');
var titleDisp = document.getElementById('leafType');
var sendbut = document.getElementById('sendButton');



socket.on('treeDisplay', function(data){ // Recieves information about tree based on leaf photo sent
    console.log(data.info);
    titleDisp.innerHTML = data.ltype
    resDisp.innerHTML = data.info;
    picDisp.src = data.picPath;
    // socket.emit('sendmeafile');
});

sendbut.onclick=function(){ //Send request to classify leaf
    socket.emit('leafClassify', 'Request Data');
    // $("#pic").attr('src','http://localhost:3000/myImage.png' );
    //console.log($("#myInput").val());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//Code obtained for transfering image from https://github.com/rico345100/socket.io-file-example
uploader.on('ready', function() {
	console.log('SocketIOFile ready to go!');
});
uploader.on('start', function(fileInfo) {
	console.log('Start uploading', fileInfo);
});
uploader.on('stream', function(fileInfo) {
	console.log('Streaming... sent ' + fileInfo.sent + ' bytes.');
});
uploader.on('complete', function(fileInfo) {
	document.getElementById('sendButton').style.visibility="visible";
	console.log('Upload Complete', fileInfo);
});
uploader.on('error', function(err) {
	console.log('Error!', err);
});
uploader.on('abort', function(fileInfo) {
	console.log('Aborted: ', fileInfo);
});

form.onsubmit = function(ev) { //Upload captured leaf image
	ev.preventDefault();
	
	// Send File Element to upload
	var fileEl = document.getElementById('file');
	// var uploadIds = uploader.upload(fileEl);

	// Or just pass file objects directly

	var uploadIds = uploader.upload(fileEl.files);

	// socket.emit('join', 'Request Data');
};

//////////////////////////////////////////////////////////////////////////////////////////////////////
