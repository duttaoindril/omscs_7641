const fs = require('fs');
const json = require('../yoga/yoga-master-training-db-export.json');

const config = json.config;
console.log(config);

const getPoseIndex = (pose) => {
    switch (pose) {
        case 'warriorii': return 1;
        case 'triangle': return 2;
        case 'tree': return 3;
    }
    return 0;
}

const data = Object.values(json.frames).reduce((acc, frame) => {
    acc.push([...frame.datatype4, getPoseIndex(frame.pose)]);
    return acc;
}, []);

arrayToCSV(data)

function arrayToCSV(array) {
    var csvRows = [];
    for (var i = 0; i < array.length; ++i) {
        csvRows.push(array[i].join(','));
    }
    var csvString = csvRows.join('\r\n');
    fs.writeFile('yoga.csv', csvString, function(err) {});
}