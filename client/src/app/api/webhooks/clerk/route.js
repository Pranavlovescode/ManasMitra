import { headers } from 'next/headers';
import { Webhook } from 'svix';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

const webhookSecret = process.env.CLERK_WEBHOOK_SECRET;

export async function POST(req) {
  if (!webhookSecret) {
    throw new Error('Please add CLERK_WEBHOOK_SECRET from Clerk Dashboard to .env.local');
  }

  // Get the headers
  const headerPayload = headers();
  const svix_id = headerPayload.get('svix-id');
  const svix_timestamp = headerPayload.get('svix-timestamp');
  const svix_signature = headerPayload.get('svix-signature');

  // If there are no headers, error out
  if (!svix_id || !svix_timestamp || !svix_signature) {
    return new Response('Error occured -- no svix headers', {
      status: 400,
    });
  }

  // Get the body
  const payload = await req.json();
  const body = JSON.stringify(payload);

  // Create a new Svix instance with your secret.
  const wh = new Webhook(webhookSecret);

  let evt;

  // Verify the payload with the headers
  try {
    evt = wh.verify(body, {
      'svix-id': svix_id,
      'svix-timestamp': svix_timestamp,
      'svix-signature': svix_signature,
    });
  } catch (err) {
    console.error('Error verifying webhook:', err);
    return new Response('Error occured', {
      status: 400,
    });
  }

  // Handle the webhook
  const eventType = evt.type;

  try {
    await connectDB();

    switch (eventType) {
      case 'user.created':
        await handleUserCreated(evt.data);
        break;
      case 'user.updated':
        await handleUserUpdated(evt.data);
        break;
      case 'user.deleted':
        await handleUserDeleted(evt.data);
        break;
      default:
        console.log(`Unhandled event type: ${eventType}`);
    }

    return new Response('', { status: 200 });
  } catch (error) {
    console.error('Error handling webhook:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}

async function handleUserCreated(userData) {
  try {
    const {
      id: clerkId,
      email_addresses,
      first_name,
      last_name,
      unsafe_metadata,
      image_url,
    } = userData;

    const primaryEmail = email_addresses.find(email => email.id === userData.primary_email_address_id);
    const role = unsafe_metadata?.role || 'patient';

    const newUser = new User({
      clerkId,
      email: primaryEmail?.email_address,
      firstName: first_name || '',
      lastName: last_name || '',
      role,
      avatar: image_url,
      profileComplete: false,
      isActive: true,
      lastLogin: new Date(),
    });

    await newUser.save();
    console.log(`User created in MongoDB: ${newUser.email}`);
  } catch (error) {
    console.error('Error creating user in MongoDB:', error);
    throw error;
  }
}

async function handleUserUpdated(userData) {
  try {
    const {
      id: clerkId,
      email_addresses,
      first_name,
      last_name,
      image_url,
    } = userData;

    const primaryEmail = email_addresses.find(email => email.id === userData.primary_email_address_id);

    await User.findOneAndUpdate(
      { clerkId },
      {
        email: primaryEmail?.email_address,
        firstName: first_name || '',
        lastName: last_name || '',
        avatar: image_url,
        lastLogin: new Date(),
      },
      { new: true }
    );

    console.log(`User updated in MongoDB: ${primaryEmail?.email_address}`);
  } catch (error) {
    console.error('Error updating user in MongoDB:', error);
    throw error;
  }
}

async function handleUserDeleted(userData) {
  try {
    const { id: clerkId } = userData;

    await User.findOneAndUpdate(
      { clerkId },
      { isActive: false },
      { new: true }
    );

    console.log(`User soft deleted in MongoDB: ${clerkId}`);
  } catch (error) {
    console.error('Error deleting user in MongoDB:', error);
    throw error;
  }
}