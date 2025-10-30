import mongoose from 'mongoose';

const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
  throw new Error('Please define the MONGODB_URI environment variable inside .env.local');
}

/**
 * Global is used here to maintain a cached connection across hot reloads
 * in development. This prevents connections growing exponentially
 * during API Route usage.
 */
let cached = global.mongoose;

if (!cached) {
  cached = global.mongoose = { conn: null, promise: null };
}

async function connectDB() {
  if (cached.conn) {
    return cached.conn;
  }

  if (!cached.promise) {
    const opts = {
      bufferCommands: false,
      useNewUrlParser: true,
      useUnifiedTopology: true,
      maxPoolSize: 10,
    };

    try {
      cached.promise = mongoose.connect(MONGODB_URI, opts).then((mongoose) => {
        console.log('MongoDB connected successfully');
        return mongoose;
      });
    } catch (error) {
      console.error('MongoDB connection error:', error);
      cached.promise = null;
      throw new Error(`MongoDB connection failed: ${error.message}`);
    }
  }

  try {
    cached.conn = await cached.promise;
  } catch (e) {
    console.error('Error establishing MongoDB connection:', e);
    cached.promise = null;
    throw new Error(`Failed to establish MongoDB connection: ${e.message}`);
  }

  return cached.conn;
}

export default connectDB;