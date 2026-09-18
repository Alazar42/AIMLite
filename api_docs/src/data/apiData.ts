export interface Parameter {
  name: string;
  type: string;
  required: boolean;
  defaultValue?: string;
  description: string;
  validation?: string;
  children?: Parameter[];
}

export interface ErrorCode {
  code: number;
  type: string;
  title: string;
  description: string;
  troubleshooting: string;
}

export interface NavItem {
  id: string;
  label: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path?: string;
}

export interface NavCategory {
  name: string;
  items: NavItem[];
}

export const NAVIGATION_CATEGORIES: NavCategory[] = [
  {
    name: 'Getting Started',
    items: [
      { id: 'overview', label: 'Overview' },
      { id: 'quickstart', label: 'Quickstart Guide' },
      { id: 'environments', label: 'Test vs Live Modes' },
    ],
  },
  {
    name: 'Core Concepts',
    items: [
      { id: 'authentication', label: 'Authentication' },
      { id: 'idempotency', label: 'Idempotency Keys' },
      { id: 'pagination', label: 'Pagination' },
      { id: 'rate-limits', label: 'Rate Limits' },
      { id: 'errors', label: 'Error Handling' },
    ],
  },
  {
    name: 'Payments API',
    items: [
      { id: 'payments-create', label: 'Create a Payment', method: 'POST', path: '/v1/payments/create' },
      { id: 'payments-get', label: 'Retrieve a Payment', method: 'GET', path: '/v1/payments/{id}' },
      { id: 'payments-capture', label: 'Capture a Payment', method: 'POST', path: '/v1/payments/{id}/capture' },
      { id: 'payments-refund', label: 'Refund a Payment', method: 'POST', path: '/v1/payments/{id}/refund' },
    ],
  },
  {
    name: 'Customers & Billing',
    items: [
      { id: 'customers-create', label: 'Create Customer', method: 'POST', path: '/v1/customers' },
      { id: 'customers-list', label: 'List Customers', method: 'GET', path: '/v1/customers' },
      { id: 'invoices-list', label: 'List Invoices', method: 'GET', path: '/v1/invoices' },
    ],
  },
  {
    name: 'Webhooks & Events',
    items: [
      { id: 'webhooks-subscribe', label: 'Register Webhook', method: 'POST', path: '/v1/webhooks' },
      { id: 'webhooks-events', label: 'Event Types', method: 'GET', path: '/v1/events' },
    ],
  },
];

export const PAYMENT_CREATE_DATA = {
  endpoint: '/v1/payments/create',
  method: 'POST' as const,
  title: 'Create a Payment',
  breadcrumbs: ['Resources', 'Payments', 'Create a Payment'],
  description:
    'Initiates an immediate or authorization-only payment transaction against an authorized payment source. Supports multi-currency charges, automatic idempotent re-tries via the Idempotency-Key header, and integrated fraud screening.',
  
  headerParams: [
    {
      name: 'Authorization',
      type: 'string',
      required: true,
      description: 'Standard HTTP Bearer token authentication header.',
      validation: 'Must start with "Bearer sec_test_" or "Bearer sec_live_"',
    },
    {
      name: 'Idempotency-Key',
      type: 'UUID string',
      required: false,
      description: 'Unique client-supplied UUID identifier. Prevents accidental duplicate charges if network timeouts occur.',
      validation: 'Standard RFC 4122 UUID v4 format',
    },
    {
      name: 'Content-Type',
      type: 'string',
      required: true,
      defaultValue: 'application/json',
      description: 'Declares the serialization format of the HTTP request payload.',
      validation: 'Must be "application/json"',
    },
  ] as Parameter[],

  bodyParams: [
    {
      name: 'amount',
      type: 'integer',
      required: true,
      description: 'The amount in the smallest currency unit (cents, pence, etc.). For instance, 2000 represents $20.00 USD.',
      validation: 'Minimum: 50. Maximum: 99999999.',
    },
    {
      name: 'currency',
      type: 'string',
      required: true,
      description: 'Three-letter ISO 4217 currency code formatted in lowercase or uppercase.',
      validation: 'One of "usd", "eur", "gbp", "cad", "aud", "jpy".',
    },
    {
      name: 'customer_id',
      type: 'string (UUID)',
      required: false,
      description: 'The unique identifier of an existing customer record to bind this charge to.',
      validation: 'Must match an active cus_ prefix identifier.',
    },
    {
      name: 'payment_method',
      type: 'object',
      required: true,
      description: 'Container specifying payment credential details and card or digital wallet authorization token.',
      children: [
        {
          name: 'type',
          type: 'enum string',
          required: true,
          description: 'Payment rail instrument identifier.',
          validation: '"card" | "bank_transfer" | "apple_pay" | "google_pay"',
        },
        {
          name: 'token',
          type: 'string',
          required: true,
          description: 'Client-side single-use token produced by the SDK checkout flow.',
          validation: 'Starts with "pm_tok_"',
        },
      ],
    },
    {
      name: 'capture',
      type: 'boolean',
      required: false,
      defaultValue: 'true',
      description: 'Whether to immediately settle funds. When false, funds are placed on authorization hold for up to 7 business days.',
    },
    {
      name: 'statement_descriptor',
      type: 'string',
      required: false,
      description: 'Custom descriptor rendered on the customer bank statement line item.',
      validation: 'Max 22 ASCII characters. No <, >, \, \', or " symbols.',
    },
    {
      name: 'metadata',
      type: 'key-value map',
      required: false,
      description: 'Set of up to 50 key-value string pairs useful for storing internal order references or analytics tags.',
      validation: 'Keys max 40 chars; Values max 500 chars.',
    },
  ] as Parameter[],

  errorCodes: [
    {
      code: 400,
      type: 'invalid_request_error',
      title: 'Bad Request',
      description: 'The payload structure, parameters, or data types failed schema validation.',
      troubleshooting: 'Check that amount is an integer >= 50 and that payment_method.token is a valid non-expired token.',
    },
    {
      code: 401,
      type: 'authentication_error',
      title: 'Unauthorized',
      description: 'No valid API key was provided or the secret key is revoked.',
      troubleshooting: 'Verify that your Authorization header contains Bearer sec_... without extra spaces.',
    },
    {
      code: 402,
      type: 'card_error',
      title: 'Payment Required / Declined',
      description: 'The card issuer declined authorization (insufficient funds, expired, suspected fraud).',
      troubleshooting: 'Inspect the decline_code attribute in the response body to surface user-actionable instructions.',
    },
    {
      code: 429,
      type: 'rate_limit_error',
      title: 'Too Many Requests',
      description: 'Request volume exceeded your workspace tier quota (standard tier: 100 req/sec).',
      troubleshooting: 'Implement exponential backoff and jitter algorithms. Inspect Retry-After header.',
    },
    {
      code: 500,
      type: 'api_error',
      title: 'Internal Server Error',
      description: 'An unexpected error occurred on the gateway infrastructure.',
      troubleshooting: 'Safe to retry when using Idempotency-Key. Check https://status.example.com.',
    },
  ] as ErrorCode[],

  codeSnippets: {
    curl: `curl -X POST https://api.aimlite.dev/v1/payments/create \\
  -H "Authorization: Bearer sec_test_99a8b7c6d5e4f3a2b1" \\
  -H "Idempotency-Key: 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d" \\
  -H "Content-Type: application/json" \\
  -d '{
    "amount": 2500,
    "currency": "usd",
    "customer_id": "cus_9xK2m1p0Lq",
    "payment_method": {
      "type": "card",
      "token": "pm_tok_visa_4242"
    },
    "capture": true,
    "metadata": {
      "order_id": "ord_9921",
      "cart_items": "2"
    }
  }'`,

    typescript: `import { AIMLiteClient } from '@aimlite/client';

const client = new AIMLiteClient({
  apiKey: process.env.AIMLITE_SECRET_KEY,
});

const payment = await client.payments.create({
  amount: 2500, // $25.00 USD
  currency: 'usd',
  customerId: 'cus_9xK2m1p0Lq',
  paymentMethod: {
    type: 'card',
    token: 'pm_tok_visa_4242',
  },
  capture: true,
  metadata: {
    orderId: 'ord_9921',
  },
}, {
  idempotencyKey: '9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d',
});

console.log('Payment initialized:', payment.id);`,

    python: `import os
import aimlite

client = aimlite.Client(api_key=os.environ.get("AIMLITE_SECRET_KEY"))

payment = client.payments.create(
    amount=2500,  # $25.00 USD
    currency="usd",
    customer_id="cus_9xK2m1p0Lq",
    payment_method={
        "type": "card",
        "token": "pm_tok_visa_4242",
    },
    capture=True,
    metadata={"order_id": "ord_9921"},
    idempotency_key="9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
)

print(f"Payment created successfully: {payment.id}")`,

    go: `package main

import (
	"context"
	"fmt"
	"os"

	"github.com/aimlite/aimlite-go"
)

func main() {
	client := aimlite.NewClient(os.Getenv("AIMLITE_SECRET_KEY"))

	params := &aimlite.PaymentCreateParams{
		Amount:   aimlite.Int64(2500),
		Currency: aimlite.String("usd"),
		CustomerID: aimlite.String("cus_9xK2m1p0Lq"),
		PaymentMethod: &aimlite.PaymentMethodParams{
			Type:  aimlite.String("card"),
			Token: aimlite.String("pm_tok_visa_4242"),
		},
		Capture: aimlite.Bool(true),
	}

	opts := &aimlite.RequestOptions{
		IdempotencyKey: "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
	}

	payment, err := client.Payments.Create(context.Background(), params, opts)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Payment created: %s\\n", payment.ID)
}`,
  },

  defaultPayload: JSON.stringify(
    {
      amount: 2500,
      currency: 'usd',
      customer_id: 'cus_9xK2m1p0Lq',
      payment_method: {
        type: 'card',
        token: 'pm_tok_visa_4242',
      },
      capture: true,
      metadata: {
        order_id: 'ord_9921',
        channel: 'web_checkout',
      },
    },
    null,
    2
  ),

  sampleResponse: {
    id: 'pay_99f2b1a8c7e6d5',
    object: 'payment',
    amount: 2500,
    amount_received: 2500,
    currency: 'usd',
    status: 'succeeded',
    customer_id: 'cus_9xK2m1p0Lq',
    payment_method: {
      id: 'pm_48921094',
      type: 'card',
      card: {
        brand: 'visa',
        last4: '4242',
        exp_month: 12,
        exp_year: 2028,
        country: 'US',
        funding: 'credit',
      },
    },
    captured: true,
    receipt_url: 'https://pay.aimlite.dev/receipts/pay_99f2b1a8c7e6d5',
    statement_descriptor: 'AIMLITE* SERVICES',
    metadata: {
      order_id: 'ord_9921',
      channel: 'web_checkout',
    },
    created_at: 1726315200,
    livemode: false,
  },

  sampleResponseHeaders: {
    'content-type': 'application/json; charset=utf-8',
    'x-request-id': 'req_884920194827103a',
    'x-ratelimit-limit': '100',
    'x-ratelimit-remaining': '98',
    'x-ratelimit-reset': '1726315201',
    'idempotency-status': 'processed',
    'date': 'Mon, 14 Sep 2026 11:20:40 GMT',
  },
};
