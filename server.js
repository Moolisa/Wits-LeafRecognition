"use strict";
const express = require('express');
const app = express();
const http = require('http');
const httpServer = http.Server(app);
const io = require('socket.io')(httpServer,{pingInterval: 3000000,
  pingTimeout: 3000000,}); //Increase connected time so socket does not time out and cause react native to refresh
const SocketIOFile = require('socket.io-file');
var zerorpc = require("zerorpc");
var path = require('path');

var port = process.env.PORT || 3500;

var imgLoc = 'leaf.jpg'; //Location of recieved leaf photo

////////////////////////////////////////////////////////////////////////////////////////////////////
//Set up access to http server side
app.get('/', (req, res, next) => {
	return res.sendFile(__dirname + '/client/index.html');
});

app.get('/app.js', (req, res, next) => {
	return res.sendFile(__dirname + '/client/app.js');
});

app.get('/css/style.css', (req, res, next) => {
	return res.sendFile(__dirname + '/client/css/style.css');
});

app.get('/socket.io.js', (req, res, next) => {
	return res.sendFile(__dirname + '/node_modules/socket.io-client/dist/socket.io.js');
});

app.get('/socket.io-file-client.js', (req, res, next) => {
	return res.sendFile(__dirname + '/node_modules/socket.io-file-client/socket.io-file-client.js');
});

app.get('/white-leaf-48.ico', (req, res, next) => {
	return res.sendFile(__dirname + '/client/img/white-leaf-48.ico');
});

app.get('/img/background.jpg', (req, res, next) => {
	return res.sendFile(__dirname + '/client/img/background.jpg');
});

app.get('/resImg1', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf.jpg');
});

app.get('/leaf1', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf1.jpg');
});

app.get('/leaf2', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf2.jpg');
});

app.get('/leaf3', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf3.jpg');
});

app.get('/leaf4', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf4.jpg');
});

app.get('/leaf5', (req, res, next) => {
	return res.sendFile(__dirname + '/client/leaf5.jpg');
});
////////////////////////////////////////////////////////////////////////////////////////////////////

//Initialise Connection to Phython server
var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4242");

// socket.io.timeout(-1);

//Connection to Front-end
io.on('connection', (socket) => {
	console.log('Socket connected.');

	socket.on('leafClassify', function(data) {// data is to test socket connection
    	console.log(data);

    	client.invoke("classify", imgLoc, function(error, res, more) {// Send image path to python
    		var treeType = "QQQQ";
    		var treeInfo = "Information about tree";
    		var treePath = "resImg1";
   //  		console.log(res);
   //  		if (res==1) {
   //  			treeType = "Podocarpus latifolius";
			// } else {
   //  			treeType = "Gonioma Kamassi";
			// } 
			var treeJson = JSON.parse(res);
			treeType = treeJson.treaName ;
			treeInfo = treeJson.dataInformation;
			treePath = treeJson.leafPath;
   			io.emit('treeDisplay', {ltype: treeType, info: treeInfo, picPath: treePath}); //Send resultant information to front-end
   			console.log("Leaf Classified");
		});
    });

    socket.on('pingme',function(data) {
    	io.emit('ping',"hello");
	console.log(data);
	});


// 	setInterval(() => {
//   io.emit('ping', { data: (new Date())/1});
// }, 1000);

////////////////////////////////////////////////////////////////////////////////////////////////////
//Socket transfer of image server side 
	var count = 0;
	var uploader = new SocketIOFile(socket, {
		uploadDir: 'data',							// simple directory
		chunkSize: 10240,							// default is 10240(1KB)
		transmissionDelay: 0,						// delay of each transmission, higher value saves more cpu resources, lower upload speed. default is 0(no delay)
		overwrite: true, 							// overwrite file if exists, default is true.;
		rename: 'leaf.jpg',
	});
	uploader.on('start', (fileInfo) => {
		console.log('Start uploading');
		console.log(fileInfo);
	});
	uploader.on('stream', (fileInfo) => {
		console.log(`${fileInfo.wrote} / ${fileInfo.size} byte(s)`);
	});
	uploader.on('complete', (fileInfo) => {
		console.log('Upload Complete.');
		console.log(fileInfo);
	});
	uploader.on('error', (err) => {
		console.log('Error!', err);
	});
	uploader.on('abort', (fileInfo) => {
		console.log('Aborted: ', fileInfo);
	});
});
////////////////////////////////////////////////////////////////////////////////////////////////////

httpServer.listen(port, () => {
	console.log('Server listening on port 3500');
});
