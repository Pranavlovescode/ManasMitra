import { headers } from 'next/headers';
import { Webhook } from 'svix';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

const webhookSecret = process.env.CLERK_WEBHOOK_SECRET;

export async function POST(req) {
  console.log('🚀 [WEBHOOK] Clerk webhook received at:', new Date().toISOString());
  console.log('🔧 [WEBHOOK] Environment check - webhookSecret exists:', !!webhookSecret);
  
  if (!webhookSecret) {
    console.error('❌ [WEBHOOK] CLERK_WEBHOOK_SECRET is missing from environment variables');
    throw new Error('Please add CLERK_WEBHOOK_SECRET from Clerk Dashboard to .env.local');
  }

  // Get the headers
  const headerPayload = await headers();
  const svix_id = headerPayload.get('svix-id');
  const svix_timestamp = headerPayload.get('svix-timestamp');
  const svix_signature = headerPayload.get('svix-signature');

  console.log('📋 [WEBHOOK] Headers received:');
  console.log('  - svix-id:', svix_id ? `${svix_id.substring(0, 10)}...` : 'MISSING');
  console.log('  - svix-timestamp:', svix_timestamp);
  console.log('  - svix-signature:', svix_signature ? `${svix_signature.substring(0, 20)}...` : 'MISSING');

  // If there are no headers, error out
  if (!svix_id || !svix_timestamp || !svix_signature) {
    console.error('❌ [WEBHOOK] Missing required svix headers');
    return new Response('Error occured -- no svix headers', {
      status: 400,
    });
  }

  // Get the body
  console.log('📦 [WEBHOOK] Parsing request body...');
  let payload;
  try {
    payload = await req.json();
    console.log('✅ [WEBHOOK] Body parsed successfully');
    console.log('📄 [WEBHOOK] Payload keys:', Object.keys(payload));
  } catch (error) {
    console.error('❌ [WEBHOOK] Failed to parse request body:', error);
    return new Response('Invalid JSON payload', { status: 400 });
  }
  
  const body = JSON.stringify(payload);
  console.log('📏 [WEBHOOK] Body length:', body.length);

  // Create a new Svix instance with your secret.
  console.log('🔐 [WEBHOOK] Creating Svix webhook instance...');
  const wh = new Webhook(webhookSecret);

  let evt;

  // Verify the payload with the headers
  console.log('🔍 [WEBHOOK] Verifying webhook signature...');
  try {
    evt = wh.verify(body, {
      'svix-id': svix_id,
      'svix-timestamp': svix_timestamp,
      'svix-signature': svix_signature,
    });
    console.log('✅ [WEBHOOK] Signature verification successful');
  } catch (err) {
    console.error('❌ [WEBHOOK] Signature verification failed:', err.message);
    console.error('🔧 [WEBHOOK] Verification details:', {
      bodyLength: body.length,
      timestamp: svix_timestamp,
      signaturePrefix: svix_signature?.substring(0, 20)
    });
    return new Response('Error occured', {
      status: 400,
    });
  }

  // Handle the webhook
  const eventType = evt.type;
  console.log('🎯 [WEBHOOK] Event type:', eventType);
  // console.log('📊 [WEBHOOK] Event data keys:', Object.keys(evt.data || {}));
  
  if (evt.data?.id) {
    console.log('👤 [WEBHOOK] User ID:', evt.data.id);
  }

  try {
    console.log('🔌 [WEBHOOK] Connecting to MongoDB...');
    await connectDB();
    console.log('✅ [WEBHOOK] MongoDB connection established');

    console.log('🔄 [WEBHOOK] Processing event...');
    switch (eventType) {
      case 'user.created':
        console.log('➕ [WEBHOOK] Handling user.created event');
        await handleUserCreated(evt.data);
        break;
      case 'user.updated':
        console.log('✏️ [WEBHOOK] Handling user.updated event');
        await handleUserUpdated(evt.data);
        break;
      case 'user.deleted':
        console.log('🗑️ [WEBHOOK] Handling user.deleted event');
        await handleUserDeleted(evt.data);
        break;
      default:
        console.log(`⚠️ [WEBHOOK] Unhandled event type: ${eventType}`);
        console.log('📋 [WEBHOOK] Available event data:', JSON.stringify(evt.data, null, 2));
    }

    console.log('✅ [WEBHOOK] Event processed successfully');
    return new Response('', { status: 200 });
  } catch (error) {
    console.error('❌ [WEBHOOK] Error handling webhook:', error);
    console.error('🔍 [WEBHOOK] Error stack:', error.stack);
    console.error('📋 [WEBHOOK] Event data at error:', JSON.stringify(evt.data, null, 2));
    return new Response('Internal Server Error', { status: 500 });
  }
}

async function handleUserCreated(userData) {
  console.log('👤 [USER_CREATED] Starting user creation process...');
  console.log('📋 [USER_CREATED] Raw user data:', JSON.stringify(userData, null, 2));
  
  try {
    const {
      id: clerkId,
      email_addresses,
      first_name,
      last_name,
      unsafe_metadata,
      image_url,
    } = userData;

    console.log('🔍 [USER_CREATED] Extracted data:');
    console.log('  - clerkId:', clerkId);
    console.log('  - email_addresses count:', email_addresses?.length || 0);
    console.log('  - first_name:', first_name);
    console.log('  - last_name:', last_name);
    console.log('  - unsafe_metadata:', unsafe_metadata);
    console.log('  - image_url:', image_url);

    const primaryEmail = email_addresses?.find(email => email.id === userData.primary_email_address_id);
    console.log('📧 [USER_CREATED] Primary email:', primaryEmail?.email_address);
    console.log('🔑 [USER_CREATED] Primary email ID:', userData.primary_email_address_id);
    
    const role = unsafe_metadata?.role || 'patient';
    console.log('👔 [USER_CREATED] Assigned role:', role);

    const newUserData = {
      clerkId,
      email: primaryEmail?.email_address,
      firstName: first_name || '',
      lastName: last_name || '',
      role,
      avatar: image_url,
      profileComplete: false,
      isActive: true,
      lastLogin: new Date(),
    };

    console.log('💾 [USER_CREATED] User data to save:', JSON.stringify(newUserData, null, 2));

    const newUser = new User(newUserData);
    console.log('🏗️ [USER_CREATED] User model created, attempting to save...');
    
    await newUser.save();
    console.log(`✅ [USER_CREATED] User successfully created in MongoDB: ${newUser.email}`);
    console.log('🆔 [USER_CREATED] MongoDB _id:', newUser._id);
  } catch (error) {
    console.error('❌ [USER_CREATED] Error creating user in MongoDB:', error);
    console.error('🔍 [USER_CREATED] Error details:');
    console.error('  - Name:', error.name);
    console.error('  - Message:', error.message);
    if (error.code) console.error('  - Code:', error.code);
    if (error.keyPattern) console.error('  - Key Pattern:', error.keyPattern);
    if (error.keyValue) console.error('  - Key Value:', error.keyValue);
    throw error;
  }
}

async function handleUserUpdated(userData) {
  console.log('✏️ [USER_UPDATED] Starting user update process...');
  console.log('📋 [USER_UPDATED] Raw user data:', JSON.stringify(userData, null, 2));
  
  try {
    const {
      id: clerkId,
      email_addresses,
      first_name,
      last_name,
      image_url,
    } = userData;

    console.log('🔍 [USER_UPDATED] Extracted data:');
    console.log('  - clerkId:', clerkId);
    console.log('  - email_addresses count:', email_addresses?.length || 0);
    console.log('  - first_name:', first_name);
    console.log('  - last_name:', last_name);
    console.log('  - image_url:', image_url);

    const primaryEmail = email_addresses?.find(email => email.id === userData.primary_email_address_id);
    console.log('📧 [USER_UPDATED] Primary email:', primaryEmail?.email_address);

    const updateData = {
      email: primaryEmail?.email_address,
      firstName: first_name || '',
      lastName: last_name || '',
      avatar: image_url,
      lastLogin: new Date(),
    };

    console.log('💾 [USER_UPDATED] Update data:', JSON.stringify(updateData, null, 2));
    console.log('🔍 [USER_UPDATED] Finding user with clerkId:', clerkId);

    const updatedUser = await User.findOneAndUpdate(
      { clerkId },
      updateData,
      { new: true }
    );

    if (updatedUser) {
      console.log(`✅ [USER_UPDATED] User successfully updated in MongoDB: ${primaryEmail?.email_address}`);
      console.log('🆔 [USER_UPDATED] MongoDB _id:', updatedUser._id);
    } else {
      console.log(`⚠️ [USER_UPDATED] No user found with clerkId: ${clerkId}`);
    }
  } catch (error) {
    console.error('❌ [USER_UPDATED] Error updating user in MongoDB:', error);
    console.error('🔍 [USER_UPDATED] Error details:');
    console.error('  - Name:', error.name);
    console.error('  - Message:', error.message);
    if (error.code) console.error('  - Code:', error.code);
    throw error;
  }
}

async function handleUserDeleted(userData) {
  console.log('🗑️ [USER_DELETED] Starting user deletion process...');
  console.log('📋 [USER_DELETED] Raw user data:', JSON.stringify(userData, null, 2));
  
  try {
    const { id: clerkId } = userData;
    console.log('🔍 [USER_DELETED] ClerkId to delete:', clerkId);

    console.log('💾 [USER_DELETED] Performing soft delete (setting isActive: false)...');
    const deletedUser = await User.findOneAndUpdate(
      { clerkId },
      { isActive: false },
      { new: true }
    );

    if (deletedUser) {
      console.log(`✅ [USER_DELETED] User soft deleted in MongoDB: ${clerkId}`);
      console.log('📧 [USER_DELETED] Deleted user email:', deletedUser.email);
      console.log('🆔 [USER_DELETED] MongoDB _id:', deletedUser._id);
    } else {
      console.log(`⚠️ [USER_DELETED] No user found with clerkId: ${clerkId}`);
    }
  } catch (error) {
    console.error('❌ [USER_DELETED] Error deleting user in MongoDB:', error);
    console.error('🔍 [USER_DELETED] Error details:');
    console.error('  - Name:', error.name);
    console.error('  - Message:', error.message);
    if (error.code) console.error('  - Code:', error.code);
    throw error;
  }
}