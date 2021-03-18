const faker = require('faker');

module.exports = {
  genDatasetName,
  genRecordData,
  genProperties,
  setNextPage
}

function genDatasetName(requestParams, context, ee, next) {
  context.vars.datasetName = `Neo4j load test ${faker.random.uuid()}`
  return next();
}

function genProperties(requestParams, context, ee, next) {
  requestParams.json = [
    {
      "name": "name",
      "displayName": "Name",
      "dataType": "String",
      "description": ""
    },
    {
      "name": "age",
      "displayName": "Age",
      "dataType": "Long",
      "description": ""
    },
    {
      "name": "dob",
      "displayName": "DOB",
      "dataType": "String", // TODO: this should be "Date"
      "description": ""
    },
    {
      "name": "streetAddress",
      "displayName": "Street Address",
      "dataType": "String",
      "description": ""
    },
    {
      "name": "city",
      "displayName": "City",
      "dataType": "String",
      "description": ""
    },
    {
      "name": "zip",
      "displayName": "Zip",
      "dataType": "String",
      "description": ""
    }
  ]
  return next();
}


/**
  * Create many records for upload
  */
function genRecordData(requestParams, context, ee, next) {
  requestParams.json = Array(1000).fill(1).map(function(i) {
    return {
      "values": {
        "name": faker.name.firstName(),
        "age": faker.random.number({
            'min': 1,
            'max': 100
        }),
        "dob": faker.date.past(),
        "streetAddress": faker.address.streetAddress(),
        "city": faker.address.city(),
        "zip": faker.address.zipCode()
      }
    };
  })
  return next();
}


function setNextPage(requestParams, response, context, ee, next) {
  let nextPage = context.vars['response_next_page']
  context.vars['next_page'] = nextPage || 0
  console.log(`RESPONSE :: iter = ${context.vars.$loopCount}, next = ${nextPage}`)
  return next()
}
