$.show = function(){alert(app.version,"提示")};

$.exportCurrentCanvasImage = function(filepath){
    
    var doc = app.activeDocument;

    opt = new PNGSaveOptions;
    opt.compression = 0;
    doc.saveAs(File(filepath),opt,true);

};