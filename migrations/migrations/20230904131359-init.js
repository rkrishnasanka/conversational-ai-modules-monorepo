module.exports = {
  async up(db, client) {
    // TODO write your migration here.
    // See https://github.com/seppevs/migrate-mongo/#creating-a-new-migration-script
    // Example:
    // await db.collection('albums').updateOne({artist: 'The Beatles'}, {$set: {blacklisted: true}});
    // Add a new field to the user collection called verified (boolean)
    await db.collection('Users').updateMany({}, {$set: {verified: false}});
  },

  async down(db, client) {
    // TODO write the statements to rollback your migration (if possible)
    // Example:
    // await db.collection('albums').updateOne({artist: 'The Beatles'}, {$set: {blacklisted: false}});

    // Remove the verified field from the user collection
    await db.collection('Users').updateMany({}, {$unset: {verified: ""}});
  }
};
