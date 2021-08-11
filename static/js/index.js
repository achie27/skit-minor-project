var SketchX = [], SketchY = [];
var Drawing = [];


$(document).ready(function() {
  initialize();
});

// works out the X, Y position of the click inside the canvas from the X, Y position on the page

function getPosition(mouseEvent, sigCanvas) {
    var rect = sigCanvas.getBoundingClientRect();
    return {
      X: mouseEvent.clientX - rect.left,
      Y: mouseEvent.clientY - rect.top
    };
}


function initialize() {
  // get references to the canvas element as well as the 2D drawing context
  var sigCanvas = document.getElementById("canvas");
  var context = sigCanvas.getContext("2d");
  context.strokeStyle = "#f8920f";
  context.lineJoin = "round";
  context.lineWidth = 10;

  // Add background image to canvas - remove for blank white canvas
  var background = new Image();
  background.src = "0";
  // Make sure the image is loaded first otherwise nothing will draw.
  background.onload = function() {
    context.drawImage(background, 0, 0);
  }

  // This will be defined on a TOUCH device such as iPad or Android, etc.
  var is_touch_device = 'ontouchstart' in document.documentElement;

  if (is_touch_device) {
    // create a drawer which tracks touch movements
    var drawer = {
      isDrawing: false,
      touchstart: function(coors) {
        context.beginPath();
        context.moveTo(coors.x, coors.y);
        this.isDrawing = true;
      },
      touchmove: function(coors) {
        if (this.isDrawing) {
          context.lineTo(coors.x, coors.y);
          context.stroke();
        }
      },
      touchend: function(coors) {
        if (this.isDrawing) {
          this.touchmove(coors);
          this.isDrawing = false;
        }
      }
    };

    // create a function to pass touch events and coordinates to drawer
    function draw(event) {

      // get the touch coordinates.  Using the first touch in case of multi-touch
      var coors = {
        x: event.targetTouches[0].pageX,
        y: event.targetTouches[0].pageY
      };

      // Now we need to get the offset of the canvas location
      var obj = sigCanvas;

      if (obj.offsetParent) {
        // Every time we find a new object, we add its offsetLeft and offsetTop to curleft and curtop.
        do {
          coors.x -= obj.offsetLeft;
          coors.y -= obj.offsetTop;
        }
        // The while loop can be "while (obj = obj.offsetParent)" only, which does return null
        // when null is passed back, but that creates a warning in some editors (i.e. VS2010).
        while ((obj = obj.offsetParent) != null);
      }

      // pass the coordinates to the appropriate handler
      drawer[event.type](coors);
    }

    // attach the touchstart, touchmove, touchend event listeners.
    sigCanvas.addEventListener('touchstart', draw, false);
    sigCanvas.addEventListener('touchmove', draw, false);
    sigCanvas.addEventListener('touchend', draw, false);

    // prevent elastic scrolling
    sigCanvas.addEventListener('touchmove', function(event) {
      event.preventDefault();
    }, false);
  } else {

    // start drawing when the mousedown event fires, and attach handlers to
    // draw a line to wherever the mouse moves to
    $("#canvas").mousedown(function(mouseEvent) {
      var position = getPosition(mouseEvent, sigCanvas);
      //Drawing.push(Sketch);
      //console.log(Drawing);
      context.moveTo(position.X, position.Y);
      context.beginPath();


      // attach event handlers
      $(this).mousemove(function(mouseEvent) {
        drawLine(mouseEvent, sigCanvas, context);
      }).mouseup(function(mouseEvent) {
        finishDrawing(mouseEvent, sigCanvas, context);
      }).mouseout(function(mouseEvent) {
        finishDrawing(mouseEvent, sigCanvas, context);
      });
    });

  }
}

// draws a line to the x and y coordinates of the mouse event inside
// the specified element using the specified context
function drawLine(mouseEvent, sigCanvas, context) {

  var position = getPosition(mouseEvent, sigCanvas);
	SketchX.push(position.X);
  SketchY.push(position.Y);
  context.lineTo(position.X, position.Y);
  context.stroke();
}

// draws a line from the last coordiantes in the path to the finishing
// coordinates and unbind any event handlers which need to be preceded
// by the mouse down event
function finishDrawing(mouseEvent, sigCanvas, context) {
  // draw the line to the finishing coordinates
  drawLine(mouseEvent, sigCanvas, context);
  Drawing.push([SketchX, SketchY]);
  console.log(Drawing);
  SketchX = [], SketchY = []; 
  context.closePath();

  // unbind any events which could draw
  $(sigCanvas).unbind("mousemove")
    .unbind("mouseup")
    .unbind("mouseout");
}

// Clear the canvas context using the canvas width and height
function clearCanvas() {
  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  Drawing = [];
  SketchY = [], SketchX = [];
     
}

function PredictCanvas() { 
	// $.ajax({
	//     url: "/classify",
	//     type: "POST",
 //      data : {image : Drawing},
 //      contentType: "application/json; charset=utf-8",
 //      dataType : "json"
	// });

  $.getJSON('/classify', {
      image: JSON.stringify(Drawing),
    })
      .done(function(data) {
        console.log("data is"+JSON.parse(data));
        document.getElementById('resultspan').innerText = 'This is probably ' + JSON.parse(data)['data'];
    });
	
}

